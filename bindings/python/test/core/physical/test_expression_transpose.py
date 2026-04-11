import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

TRANSPOSE_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.fp32,
]


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)

    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got32, expected32, rtol=1e-4, atol=1e-5)
    elif dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got32, expected32, rtol=5e-2, atol=5e-2)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _copy_to_host(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", TRANSPOSE_DTYPES)
def test_transpose_rectangular_numerical(dtype: thor.DataType):
    x = ex.input("x")
    eq = ex.compile(x.transpose(), device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, -2.0, 0.5, 3.0, -1.5], [2.5, 0.25, -4.0, 1.25, 5.0], [-3.5, 2.0, 1.5, -0.75, 0.125]],
        dtype=np.float32,
    ).astype(storage_dtype)
    expected = x_np.astype(np.float32).T

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq.output_shape(inputs_gpu) == [5, 3]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected.astype(storage_dtype), dtype)


@pytest.mark.cuda
def test_transpose_fp16_even_dims_fast_path_numerical():
    dtype = thor.DataType.fp16
    x = ex.input("x")
    eq = ex.compile(x.transpose(), device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.arange(8 * 10, dtype=np.float32).reshape(8, 10).astype(storage_dtype)
    expected = x_np.astype(np.float32).T

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected.astype(storage_dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", TRANSPOSE_DTYPES)
def test_transpose_followed_by_pointwise_numerical(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.exp(x.transpose() * 0.25 + 0.5)
    eq = ex.compile(expr, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, -2.0, 0.5, 3.0], [2.5, 0.25, -4.0, 1.25], [-3.5, 2.0, 1.5, -0.75]],
        dtype=np.float32,
    ).astype(storage_dtype)
    expected = np.exp(x_np.astype(np.float32).T * 0.25 + 0.5)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq.output_shape(inputs_gpu) == [4, 3]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected.astype(storage_dtype), dtype)
