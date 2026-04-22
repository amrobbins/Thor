import numpy as np
import pytest

import thor
from thor.physical import DeviceType, Placement
from thor.physical import DynamicExpression, DynamicExpressionBuild
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream

FLOAT_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp32,
]


def _numpy_storage_dtype(dtype: thor.DataType):
    return thor.physical.numpy_dtypes.from_thor(dtype)


def _numpy_compute_dtype(dtype: thor.DataType):
    if dtype == thor.DataType.fp8_e4m3 or dtype == thor.DataType.fp8_e5m2:
        return thor.physical.numpy_dtypes.bf16
    return _numpy_storage_dtype(dtype)


def _rtol_atol(dtype: thor.DataType) -> tuple[float, float]:
    if dtype == thor.DataType.fp32:
        return 3e-5, 3e-6
    if dtype == thor.DataType.fp16:
        return 5e-3, 5e-3
    if dtype == thor.DataType.bf16:
        return 2e-2, 2e-2
    if dtype == thor.DataType.fp8_e4m3:
        return 1.5e-1, 1.5e-1
    if dtype == thor.DataType.fp8_e5m2:
        return 2.5e-1, 2.5e-1
    raise AssertionError(f"Unhandled dtype: {dtype}")


def _make_host_and_gpu_tensor(
    dtype: thor.DataType,
    dimensions: list[int],
    gpu_num: int,
):
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu)
    descriptor = PhysicalTensor.Descriptor(dtype, dimensions=dimensions)

    host = PhysicalTensor(cpu_placement, descriptor)
    gpu = PhysicalTensor(gpu_placement, descriptor)
    return host, gpu


def _copy_numpy_to_gpu(
    np_array: np.ndarray,
    dtype: thor.DataType,
    gpu_num: int,
    stream: Stream,
) -> tuple[PhysicalTensor, PhysicalTensor]:
    host, gpu = _make_host_and_gpu_tensor(dtype, list(np_array.shape), gpu_num)

    host_np = host.numpy()
    host_np[:] = np_array.astype(_numpy_storage_dtype(dtype))

    gpu.copy_from_async(host, stream)
    return host, gpu


def _copy_gpu_to_cpu_numpy(
    gpu_tensor: PhysicalTensor,
    dtype: thor.DataType,
    stream: Stream,
) -> np.ndarray:
    cpu_placement = Placement(DeviceType.cpu)
    out_cpu = gpu_tensor.clone(cpu_placement)
    out_cpu.copy_from_async(gpu_tensor, stream)
    stream.synchronize()
    return np.array(out_cpu.numpy(), copy=True)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dynamic_expression_single_output_numerical(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x_init = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
        dtype=thor.physical.numpy_dtypes.fp32,
    )
    y_init = np.array(
        [[1.1, 1.2, 1.3, 1.4], [1.5, 1.2, 1.1, 1.3]],
        dtype=thor.physical.numpy_dtypes.fp32,
    )

    _, x_gpu = _copy_numpy_to_gpu(x_init, dtype, gpu_num, stream)
    _, y_gpu = _copy_numpy_to_gpu(y_init, dtype, gpu_num, stream)

    def builder(inputs, outputs, stream):
        x = ex.input("x")
        y = ex.input("y")
        expr = ex.sqrt((x + 1.5) * (y + 2.0))
        fused_equation = ex.compile(expr, device_num=gpu_num, use_fast_math=False)
        return DynamicExpressionBuild(
            equation=fused_equation,
            stamp_inputs=inputs,
        )

    dyn = DynamicExpression(builder)

    stamped = dyn.stamp(
        {
            "x": x_gpu,
            "y": y_gpu,
        },
        {},
        stream,
    )
    stamped.run()

    compute_np_dtype = _numpy_compute_dtype(dtype)
    x_ref = x_init.astype(compute_np_dtype)
    y_ref = y_init.astype(compute_np_dtype)
    expected = np.sqrt((x_ref + 1.5) * (y_ref + 2.0)).astype(_numpy_storage_dtype(dtype))

    out_np = _copy_gpu_to_cpu_numpy(stamped.output(), dtype, stream)

    rtol, atol = _rtol_atol(dtype)
    np.testing.assert_allclose(out_np, expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dynamic_expression_runtime_scalar_override_numerical(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x_init = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=thor.physical.numpy_dtypes.fp32,
    )
    y_init = np.array(
        [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
        dtype=thor.physical.numpy_dtypes.fp32,
    )

    _, x_gpu = _copy_numpy_to_gpu(x_init, dtype, gpu_num, stream)
    _, y_gpu = _copy_numpy_to_gpu(y_init, dtype, gpu_num, stream)

    def builder(inputs, outputs, stream):
        x = ex.input("x")
        y = ex.input("y")
        scale = ex.runtime_scalar("scale")
        expr = x + scale * y
        fused_equation = ex.compile(expr, device_num=gpu_num, use_fast_math=False)
        return DynamicExpressionBuild(
            equation=fused_equation,
            stamp_inputs=inputs,
        )

    dyn = DynamicExpression(builder)
    stamped = dyn.stamp(
        {
            "x": x_gpu,
            "y": y_gpu,
        },
        {},
        stream,
    )

    stamped.run({
        "scale": 0.25,
    })
    out_np = _copy_gpu_to_cpu_numpy(stamped.output(), dtype, stream)

    compute_np_dtype = _numpy_compute_dtype(dtype)
    expected = (
        x_init.astype(compute_np_dtype) +
        np.array(0.25, dtype=compute_np_dtype) * y_init.astype(compute_np_dtype)).astype(_numpy_storage_dtype(dtype))

    rtol, atol = _rtol_atol(dtype)
    np.testing.assert_allclose(out_np, expected, rtol=rtol, atol=atol)

    stamped.run({
        "scale": 2.0,
    })
    out_np = _copy_gpu_to_cpu_numpy(stamped.output(), dtype, stream)

    expected = (
        x_init.astype(compute_np_dtype) +
        np.array(2.0, dtype=compute_np_dtype) * y_init.astype(compute_np_dtype)).astype(_numpy_storage_dtype(dtype))

    np.testing.assert_allclose(out_np, expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dynamic_expression_multi_output_numerical(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x_init = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=thor.physical.numpy_dtypes.fp32,
    )
    y_init = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dtype=thor.physical.numpy_dtypes.fp32,
    )

    _, x_gpu = _copy_numpy_to_gpu(x_init, dtype, gpu_num, stream)
    _, y_gpu = _copy_numpy_to_gpu(y_init, dtype, gpu_num, stream)

    def builder(inputs, outputs, stream):
        x = ex.input("x")
        y = ex.input("y")
        outs = ex.outputs({
            "sum": x + y,
            "prod": x * y,
        })
        fused_equation = ex.compile(outs, device_num=gpu_num, use_fast_math=False)
        return DynamicExpressionBuild(
            equation=fused_equation,
            stamp_inputs=inputs,
        )

    dyn = DynamicExpression(builder)
    stamped = dyn.stamp(
        {
            "x": x_gpu,
            "y": y_gpu,
        },
        {},
        stream,
    )
    stamped.run()

    compute_np_dtype = _numpy_compute_dtype(dtype)
    x_ref = x_init.astype(compute_np_dtype)
    y_ref = y_init.astype(compute_np_dtype)

    expected_sum = (x_ref + y_ref).astype(_numpy_storage_dtype(dtype))
    expected_prod = (x_ref * y_ref).astype(_numpy_storage_dtype(dtype))

    sum_np = _copy_gpu_to_cpu_numpy(stamped.output("sum"), dtype, stream)
    prod_np = _copy_gpu_to_cpu_numpy(stamped.output("prod"), dtype, stream)

    rtol, atol = _rtol_atol(dtype)
    np.testing.assert_allclose(sum_np, expected_sum, rtol=rtol, atol=atol)
    np.testing.assert_allclose(prod_np, expected_prod, rtol=rtol, atol=atol)

    assert set(stamped.output_names()) == {"sum", "prod"}


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dynamic_expression_builder_receives_validated_inputs(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x_init = np.array([[1.0, 2.0]], dtype=thor.physical.numpy_dtypes.fp32)
    y_init = np.array([[3.0, 4.0]], dtype=thor.physical.numpy_dtypes.fp32)

    _, x_gpu = _copy_numpy_to_gpu(x_init, dtype, gpu_num, stream)
    _, y_gpu = _copy_numpy_to_gpu(y_init, dtype, gpu_num, stream)

    seen = {
        "called": False,
        "keys": None,
        "gpu_num": None,
    }

    def builder(inputs, outputs, stream):
        seen["called"] = True
        seen["keys"] = sorted(inputs.keys())
        seen["gpu_num"] = stream.get_gpu_num()

        expr = ex.input("x") - ex.input("y")
        fused_equation = ex.compile(expr, device_num=gpu_num, use_fast_math=False)
        return DynamicExpressionBuild(
            equation=fused_equation,
            stamp_inputs=inputs,
        )

    dyn = DynamicExpression(builder)
    stamped = dyn.stamp(
        {
            "x": x_gpu,
            "y": y_gpu,
        },
        {},
        stream,
    )
    stamped.run()

    assert seen["called"] is True
    assert seen["keys"] == ["x", "y"]
    assert seen["gpu_num"] == gpu_num


@pytest.mark.cuda
def test_dynamic_expression_empty_inputs_raises():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    def builder(inputs, stream):
        raise AssertionError("builder should not be called when inputs are empty")

    dyn = DynamicExpression(builder)

    with pytest.raises(ValueError, match="at least one input tensor"):
        dyn.stamp({}, {}, stream)
