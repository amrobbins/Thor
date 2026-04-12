import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _conv2d_nchw_ref(
        x: np.ndarray,
        w: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0) -> np.ndarray:
    """Reference that matches Thor's existing GpuConvolution path: true convolution (kernel flipped in spatial dims)."""
    x32 = x.astype(np.float32)
    w32 = w.astype(np.float32)

    n, c, h, width = x32.shape
    k, c2, r, s = w32.shape
    assert c == c2

    out_h = (h + 2 * pad_h - r) // stride_h + 1
    out_w = (width + 2 * pad_w - s) // stride_w + 1

    xpad = np.pad(x32, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    wflip = np.flip(w32, axis=(2, 3))

    out = np.zeros((n, k, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for ko in range(k):
            for oh in range(out_h):
                ih = oh * stride_h
                for ow in range(out_w):
                    iw = ow * stride_w
                    window = xpad[ni, :, ih:ih + r, iw:iw + s]
                    out[ni, ko, oh, ow] = np.sum(window * wflip[ko])
    return out


@pytest.mark.cuda
def test_conv2d_forward_numerical_fp16():
    dtype = thor.DataType.fp16
    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(x, w, stride_h=2, stride_w=1, pad_h=1, pad_w=0)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = (np.arange(2 * 3 * 5 * 4, dtype=np.float32).reshape(2, 3, 5, 4) / 50.0).astype(storage_dtype)
    w_np = (np.arange(4 * 3 * 3 * 2, dtype=np.float32).reshape(4, 3, 3, 2) / 40.0).astype(storage_dtype)
    expected = _conv2d_nchw_ref(x_np, w_np, stride_h=2, stride_w=1, pad_h=1, pad_w=0)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(got.astype(np.float32), expected.astype(np.float32), rtol=5e-2, atol=5e-2)


@pytest.mark.cuda
def test_conv2d_followed_by_pointwise_numerical_fp16():
    dtype = thor.DataType.fp16
    x = ex.input("x")
    w = ex.input("w")
    b = ex.input("b")

    out = ex.exp(ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1) + b)
    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = ((np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4) - 8.0) / 20.0).astype(storage_dtype)
    w_np = ((np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3) - 10.0) / 30.0).astype(storage_dtype)
    b_np = np.array([0.1, -0.2, 0.05], dtype=np.float32).reshape(1, 3, 1, 1).astype(storage_dtype)

    conv_ref = _conv2d_nchw_ref(x_np, w_np, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    # Match Thor's fp16 output storage semantics: exp is computed into an fp16 tensor,
    # so values above the finite fp16 range overflow to +inf before we copy back to fp32.
    expected = np.exp(conv_ref + b_np.astype(np.float32)).astype(storage_dtype).astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(got.astype(np.float32), expected.astype(np.float32), rtol=7e-2, atol=7e-2)


@pytest.mark.cuda
def test_conv2d_parameter_fan_override_filter():
    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    eq = ex.compile(out, device_num=0)

    dtype = thor.DataType.fp16
    stream = Stream(gpu_num=0)
    x_np = np.zeros((2, 3, 8, 7), dtype=np.float16)
    w_np = np.zeros((5, 3, 3, 2), dtype=np.float16)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
    }

    fan = eq.get_parameter_fan_overrides(inputs_gpu, ["w"])
    assert fan["w"]["fan_in"] == 3 * 3 * 2
    assert fan["w"]["fan_out"] == 5 * 3 * 2


def test_conv2d_backward_rejected_for_now():
    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    eq = ex.compile(out, device_num=0)

    with pytest.raises(RuntimeError, match="CONV2D"):
        eq.compile_backward(["x", "w"])
