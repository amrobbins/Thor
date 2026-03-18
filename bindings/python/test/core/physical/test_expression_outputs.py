import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType

FLOAT_DTYPES = [
    thor.DataType.fp32,
    # enable later once multi-output path supports them end-to-end
    # thor.DataType.fp16,
    # thor.DataType.bf16,
    # thor.DataType.fp8_e4m3,
    # thor.DataType.fp8_e5m2,
]


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return thor.physical.numpy_dtypes.from_thor(dtype)


def _numpy_compute_dtype(dtype: thor.DataType) -> np.dtype:
    if dtype == thor.DataType.fp32:
        return thor.physical.numpy_dtypes.fp32
    if dtype == thor.DataType.fp16:
        return thor.physical.numpy_dtypes.fp32
    if dtype == thor.DataType.bf16:
        return thor.physical.numpy_dtypes.fp32
    if dtype == thor.DataType.fp8_e4m3:
        return thor.physical.numpy_dtypes.fp32
    if dtype == thor.DataType.fp8_e5m2:
        return thor.physical.numpy_dtypes.fp32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _copy_numpy_to_gpu(arr: np.ndarray, dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    cpu = _cpu_tensor(list(arr.shape), dtype)
    gpu = _gpu_tensor(list(arr.shape), dtype, gpu_num=gpu_num)
    np.copyto(cpu.numpy(), arr)
    stream = Stream(gpu_num=gpu_num)
    gpu.copy_from_async(cpu, stream)
    stream.synchronize()
    return gpu


def _copy_gpu_to_numpy(t: PhysicalTensor, gpu_num: int = 0) -> np.ndarray:
    cpu = _cpu_tensor(list(t.get_dimensions()), t.get_descriptor().get_data_type())
    stream = Stream(gpu_num=gpu_num)
    cpu.copy_from_async(t, stream)
    stream.synchronize()
    return cpu.numpy().copy()


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_allclose(got, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.cuda
def test_outputs_compile_smoke():
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    compiled = outs.compile(
        dtype=thor.DataType.fp32,
        device_num=0,
        use_fast_math=False,
    )

    assert compiled is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_two_pointwise_results_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 2, 3, 4], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([5, 6, 7, 8], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected_sum = x_ref + y_ref
    expected_prod = x_ref * y_ref

    x_gpu = _copy_numpy_to_gpu(x_np, dtype)
    y_gpu = _copy_numpy_to_gpu(y_np, dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    cpu_placement = Placement(DeviceType.cpu, 0)
    sum_cpu = stamped.output("sum").clone(cpu_placement)
    prod_cpu = stamped.output("prod").clone(cpu_placement)

    sum_cpu.copy_from_async(stamped.output("sum"), stream)
    prod_cpu.copy_from_async(stamped.output("prod"), stream)
    stream.synchronize()

    got_sum = sum_cpu.numpy()
    got_prod = prod_cpu.numpy()

    _assert_close(got_sum, expected_sum, dtype)
    _assert_close(got_prod, expected_prod, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_output_stamp_missing_input_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 2, 3, 4], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([5, 6, 7, 8], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected_sum = x_ref + y_ref
    expected_prod = x_ref * y_ref

    x_gpu = _copy_numpy_to_gpu(x_np, dtype)
    y_gpu = _copy_numpy_to_gpu(y_np, dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)

    with pytest.raises(RuntimeError):
        stamped = eq.stamp({
            "x": x_gpu,
        }, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_output_stamp_unexpected_input_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 2, 3, 4], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([5, 6, 7, 8], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected_sum = x_ref + y_ref
    expected_prod = x_ref * y_ref

    x_gpu = _copy_numpy_to_gpu(x_np, dtype)
    y_gpu = _copy_numpy_to_gpu(y_np, dtype)
    z_gpu = _copy_numpy_to_gpu(y_np, dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)

    with pytest.raises(RuntimeError):
        stamped = eq.stamp({
            "x": x_gpu,
            "y": y_gpu,
            "z": z_gpu,
        }, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_stamp_wrong_input_name_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 2, 3, 4], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([5, 6, 7, 8], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected_sum = x_ref + y_ref
    expected_prod = x_ref * y_ref

    x_gpu = _copy_numpy_to_gpu(x_np, dtype)
    y_gpu = _copy_numpy_to_gpu(y_np, dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)

    with pytest.raises(RuntimeError):
        stamped = eq.stamp({
            "x": x_gpu,
            "b": y_gpu
        }, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_shared_trunk_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    trunk = (x + y) * 2.0

    outs = ex.outputs({
        "plus_one": trunk + 1.0,
        "minus_three": trunk - 3.0,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 2, 3, 4], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([2, 3, 4, 5], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)
    trunk_ref = (x_ref + y_ref) * 2.0

    expected_plus_one = trunk_ref + 1.0
    expected_minus_three = trunk_ref - 3.0

    x_gpu = _copy_numpy_to_gpu(x_np, dtype)
    y_gpu = _copy_numpy_to_gpu(y_np, dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    cpu_placement = Placement(DeviceType.cpu, 0)
    plus_one_cpu = stamped.output("plus_one").clone(cpu_placement)
    minus_three_cpu = stamped.output("minus_three").clone(cpu_placement)

    plus_one_cpu.copy_from_async(stamped.output("plus_one"), stream)
    minus_three_cpu.copy_from_async(stamped.output("minus_three"), stream)
    stream.synchronize()

    got_plus_one = plus_one_cpu.numpy()
    got_minus_three = minus_three_cpu.numpy()

    _assert_close(got_plus_one, expected_plus_one, dtype)
    _assert_close(got_minus_three, expected_minus_three, dtype)

    # Since multiple outputs, it is not valid to call .output_tensor
    with pytest.raises(RuntimeError):
        stamped.output_tensor


@pytest.mark.cuda
def test_outputs_unknown_name_rejected_after_run():
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
    })

    x_np = np.array([1, 2, 3], dtype=np.float32)
    y_np = np.array([4, 5, 6], dtype=np.float32)

    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32)
    y_gpu = _copy_numpy_to_gpu(y_np, thor.DataType.fp32)

    eq = outs.compile(dtype=thor.DataType.fp32, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()
    stream.synchronize()

    with pytest.raises(RuntimeError):
        stamped.output("does_not_exist")


@pytest.mark.cuda
def test_outputs_single_output_supports_output_tensor():
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
    })

    x_np = np.array([1, 2, 3], dtype=np.float32)
    y_np = np.array([4, 5, 6], dtype=np.float32)

    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32)
    y_gpu = _copy_numpy_to_gpu(y_np, thor.DataType.fp32)

    eq = outs.compile(dtype=thor.DataType.fp32, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()
    stream.synchronize()

    # Does not raise
    assert isinstance(stamped.output_tensor, PhysicalTensor)


@pytest.mark.cuda
def test_outputs_multiple_reductions_from_shared_trunk():
    x = ex.input("x")
    y = ex.input("y")

    trunk = x + y

    outs = ex.outputs(
        {
            "sum0": ex.reduce_sum(trunk, axis=0, squeeze=False),
            "max1": ex.reduce_max(trunk, axis=1, squeeze=False),
        })

    x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    trunk_ref = x_np + y_np

    expected_sum0 = trunk_ref.sum(axis=0, keepdims=True)
    expected_max1 = trunk_ref.max(axis=1, keepdims=True)

    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32)
    y_gpu = _copy_numpy_to_gpu(y_np, thor.DataType.fp32)

    eq = outs.compile(dtype=thor.DataType.fp32, device_num=0, use_fast_math=False)
    stream = Stream(gpu_num=0)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    cpu_placement = Placement(DeviceType.cpu, 0)
    sum0_cpu = stamped.output("sum0").clone(cpu_placement)
    max1_cpu = stamped.output("max1").clone(cpu_placement)

    sum0_cpu.copy_from_async(stamped.output("sum0"), stream)
    max1_cpu.copy_from_async(stamped.output("max1"), stream)

    stream.synchronize()

    got_sum0 = sum0_cpu.numpy()
    got_max1 = max1_cpu.numpy()

    np.testing.assert_allclose(got_sum0, expected_sum0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_max1, expected_max1, rtol=1e-6, atol=1e-6)
