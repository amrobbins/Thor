import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

FLOAT_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp32,
]


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _copy_array_to_gpu_tensor(dst: PhysicalTensor, arr: np.ndarray, dtype: thor.DataType, stream: Stream):
    host = _cpu_tensor(list(arr.shape), dtype)
    host.numpy()[...] = arr
    dst.copy_from_async(host, stream)


def _tensor_scalar_binding(buffer: PhysicalTensor, source_dtype: thor.DataType, byte_offset: int = 0):
    binding = thor.physical.TensorScalarBinding()
    binding.buffer = buffer
    binding.byte_offset = byte_offset
    binding.source_dtype = source_dtype
    return binding


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)

    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got32, expected32, rtol=1e-5, atol=1e-6)
    elif dtype in (thor.DataType.fp16, thor.DataType.bf16):
        np.testing.assert_allclose(got32, expected32, rtol=5e-3, atol=5e-3)
    elif dtype in (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2):
        np.testing.assert_allclose(got32, expected32, rtol=1e-1, atol=1e-1)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


def _gpu_to_numpy(src: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(src.dimensions), dtype)
    host.copy_from_async(src, stream)
    stream.synchronize()
    return host.numpy().copy()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_tensor_runtime_scalar_single_output_stamped_reads_gpu_buffer_and_byte_offset_numerical(dtype: thor.DataType):
    x = ex.input("x")
    step = ex.tensor_runtime_scalar("step")

    out = (x * step) + ex.constant_scalar(1.0)
    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.5]], dtype=np.float32).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)

    # Bind to the second fp32 in the buffer to exercise byte_offset handling.
    step_buffer_gpu = _host_to_gpu(np.array([123.0, 0.25], dtype=np.float32), thor.DataType.fp32, stream)
    binding = _tensor_scalar_binding(step_buffer_gpu, thor.DataType.fp32, byte_offset=4)

    stamped = eq.stamp(
        {
            "x": x_gpu,
        },
        stream,
        tensor_scalar_inputs={
            "step": binding,
        },
    )

    stamped.run()
    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    expected = ((x_np.astype(np.float32) * 0.25) + 1.0).astype(storage_dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)

    # Update the already-bound GPU buffer and verify the stamped plan reads the new value.
    _copy_array_to_gpu_tensor(step_buffer_gpu, np.array([123.0, -1.5], dtype=np.float32), thor.DataType.fp32, stream)

    stamped.run()
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    expected = ((x_np.astype(np.float32) * -1.5) + 1.0).astype(storage_dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_tensor_runtime_scalar_multi_output_stamped_with_preallocated_outputs_numerical(dtype: thor.DataType):
    x = ex.input("x")
    step = ex.tensor_runtime_scalar("step")

    outs = ex.outputs({
        "scaled": x * step,
        "shifted": x + step,
    })
    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]], dtype=np.float32).astype(storage_dtype)
    step_value = -1.5

    expected = {
        "scaled": (x_np.astype(np.float32) * step_value).astype(storage_dtype),
        "shifted": (x_np.astype(np.float32) + step_value).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)
    step_buffer_gpu = _host_to_gpu(np.array([step_value], dtype=np.float32), thor.DataType.fp32, stream)
    binding = _tensor_scalar_binding(step_buffer_gpu, thor.DataType.fp32)

    outputs_gpu = {
        "scaled": _gpu_tensor(list(x_np.shape), dtype, gpu_num=0),
        "shifted": _gpu_tensor(list(x_np.shape), dtype, gpu_num=0),
    }

    stamped = eq.stamp(
        {
            "x": x_gpu,
        },
        stream,
        tensor_scalar_inputs={
            "step": binding,
        },
        preallocated_outputs=outputs_gpu,
    )

    stamped.run()

    assert set(stamped.output_names()) == {"scaled", "shifted"}

    for name in stamped.output_names():
        out_gpu = stamped.output(name)
        out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
        out_host.copy_from_async(out_gpu, stream)
        stream.synchronize()
        got = out_host.numpy().copy()

        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_runtime_scalars_remain_distinct_in_shared_fused_stage_numerical(dtype: thor.DataType):
    x = ex.input("x")
    alpha = ex.runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")

    outs = ex.outputs({
        "affine": x * alpha + beta,
        "swapped": x * beta + alpha,
        "combo": (x + alpha) * (x + beta),
    })
    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, -2.0, 3.5], [4.0, 0.5, -6.0]], dtype=np.float32).astype(storage_dtype)
    alpha_value = 0.25
    beta_value = -1.5

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)

    assert eq._debug_stage_kinds({
        "x": x_gpu
    }) == ["FusedKernel"]

    stamped = eq.stamp({
        "x": x_gpu
    }, stream)
    stamped.run({
        "alpha": alpha_value,
        "beta": beta_value
    })

    expected = {
        "affine": (x_np.astype(np.float32) * alpha_value + beta_value).astype(storage_dtype),
        "swapped": (x_np.astype(np.float32) * beta_value + alpha_value).astype(storage_dtype),
        "combo":
            ((x_np.astype(np.float32) + alpha_value) * (x_np.astype(np.float32) + beta_value)).astype(storage_dtype),
    }

    for name in stamped.output_names():
        got = _gpu_to_numpy(stamped.output(name), dtype, stream)
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_tensor_runtime_scalars_remain_distinct_in_shared_fused_stage_numerical(dtype: thor.DataType):
    x = ex.input("x")
    alpha = ex.tensor_runtime_scalar("alpha")
    beta = ex.tensor_runtime_scalar("beta")

    outs = ex.outputs({
        "affine": x * alpha + beta,
        "swapped": x * beta + alpha,
        "combo": (x + alpha) * (x + beta),
    })
    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.5, -2.0, 3.0], [4.25, 0.5, -6.5]], dtype=np.float32).astype(storage_dtype)
    alpha_value = 0.75
    beta_value = -0.5

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)
    alpha_buffer_gpu = _host_to_gpu(np.array([111.0, alpha_value], dtype=np.float32), thor.DataType.fp32, stream)
    beta_buffer_gpu = _host_to_gpu(np.array([beta_value], dtype=np.float32), thor.DataType.fp32, stream)

    tensor_scalar_inputs = {
        "alpha": _tensor_scalar_binding(alpha_buffer_gpu, thor.DataType.fp32, byte_offset=4),
        "beta": _tensor_scalar_binding(beta_buffer_gpu, thor.DataType.fp32),
    }

    assert eq._debug_stage_kinds({
        "x": x_gpu
    }, tensor_scalar_inputs=tensor_scalar_inputs) == ["FusedKernel"]

    stamped = eq.stamp(
        {
            "x": x_gpu
        },
        stream,
        tensor_scalar_inputs=tensor_scalar_inputs,
    )
    stamped.run()

    expected = {
        "affine": (x_np.astype(np.float32) * alpha_value + beta_value).astype(storage_dtype),
        "swapped": (x_np.astype(np.float32) * beta_value + alpha_value).astype(storage_dtype),
        "combo":
            ((x_np.astype(np.float32) + alpha_value) * (x_np.astype(np.float32) + beta_value)).astype(storage_dtype),
    }

    for name in stamped.output_names():
        got = _gpu_to_numpy(stamped.output(name), dtype, stream)
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_mixed_runtime_scalar_kinds_share_single_fused_stage_numerical(dtype: thor.DataType):
    x = ex.input("x")
    alpha = ex.runtime_scalar("alpha")
    beta = ex.tensor_runtime_scalar("beta")
    gamma = ex.runtime_scalar("gamma")

    outs = ex.outputs({
        "main": (x * alpha) + beta + gamma,
        "mirror": (x * beta) + alpha - gamma,
    })
    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[2.0, -1.0, 0.5], [3.0, -4.0, 1.5]], dtype=np.float32).astype(storage_dtype)
    alpha_value = 1.25
    beta_value = -0.75
    gamma_value = 0.5

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)
    beta_buffer_gpu = _host_to_gpu(np.array([beta_value], dtype=np.float32), thor.DataType.fp32, stream)

    tensor_scalar_inputs = {
        "beta": _tensor_scalar_binding(beta_buffer_gpu, thor.DataType.fp32),
    }

    assert eq._debug_stage_kinds({
        "x": x_gpu
    }, tensor_scalar_inputs=tensor_scalar_inputs) == ["FusedKernel"]

    stamped = eq.stamp(
        {
            "x": x_gpu
        },
        stream,
        tensor_scalar_inputs=tensor_scalar_inputs,
    )
    stamped.run({
        "alpha": alpha_value,
        "gamma": gamma_value
    })

    expected = {
        "main": (x_np.astype(np.float32) * alpha_value + beta_value + gamma_value).astype(storage_dtype),
        "mirror": (x_np.astype(np.float32) * beta_value + alpha_value - gamma_value).astype(storage_dtype),
    }

    for name in stamped.output_names():
        got = _gpu_to_numpy(stamped.output(name), dtype, stream)
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
def test_tensor_runtime_scalar_missing_input_rejected_at_stamp():
    x = ex.input("x")
    step = ex.tensor_runtime_scalar("step")
    eq = ex.compile(x * step, device_num=0)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, thor.DataType.fp32, stream)

    with pytest.raises(RuntimeError, match="Missing required fused equation tensor runtime scalar input: step"):
        eq.stamp(
            {
                "x": x_gpu,
            },
            stream,
        )


@pytest.mark.cuda
def test_tensor_runtime_scalar_unexpected_input_rejected_at_stamp():
    x = ex.input("x")
    step = ex.tensor_runtime_scalar("step")
    eq = ex.compile(x * step, device_num=0)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, thor.DataType.fp32, stream)

    step_buffer_gpu = _host_to_gpu(np.array([0.25], dtype=np.float32), thor.DataType.fp32, stream)
    extra_buffer_gpu = _host_to_gpu(np.array([7.0], dtype=np.float32), thor.DataType.fp32, stream)

    with pytest.raises(RuntimeError, match="Unexpected tensor runtime scalar input sent to fused equation: extra"):
        eq.stamp(
            {
                "x": x_gpu,
            },
            stream,
            tensor_scalar_inputs={
                "step": _tensor_scalar_binding(step_buffer_gpu, thor.DataType.fp32),
                "extra": _tensor_scalar_binding(extra_buffer_gpu, thor.DataType.fp32),
            },
        )


@pytest.mark.cuda
def test_tensor_runtime_scalar_cpu_buffer_rejected_at_stamp():
    x = ex.input("x")
    step = ex.tensor_runtime_scalar("step")
    eq = ex.compile(x * step, device_num=0)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, thor.DataType.fp32, stream)

    step_buffer_cpu = _cpu_tensor([1], thor.DataType.fp32)
    step_buffer_cpu.numpy()[0] = 0.25

    with pytest.raises(RuntimeError, match="Tensor runtime scalar buffer must be on GPU"):
        eq.stamp(
            {
                "x": x_gpu,
            },
            stream,
            tensor_scalar_inputs={
                "step": _tensor_scalar_binding(step_buffer_cpu, thor.DataType.fp32),
            },
        )


@pytest.mark.cuda
def test_tensor_runtime_scalar_binding_bounds_checked_at_stamp():
    x = ex.input("x")
    step = ex.tensor_runtime_scalar("step")
    eq = ex.compile(x * step, device_num=0)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, thor.DataType.fp32, stream)

    step_buffer_gpu = _host_to_gpu(np.array([0.25], dtype=np.float32), thor.DataType.fp32, stream)

    with pytest.raises(RuntimeError, match="Tensor runtime scalar binding exceeds backing buffer size"):
        eq.stamp(
            {
                "x": x_gpu,
            },
            stream,
            tensor_scalar_inputs={
                "step": _tensor_scalar_binding(step_buffer_gpu, thor.DataType.fp32, byte_offset=4),
            },
        )
