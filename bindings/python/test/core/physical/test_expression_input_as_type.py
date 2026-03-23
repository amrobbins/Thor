# bindings/python/test/core/physical/test_expression_input_as_type.py

import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType, numpy_dtypes


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _numpy_compute_dtype(dtype: thor.DataType) -> np.dtype:
    if dtype == thor.DataType.fp8_e4m3:
        return numpy_dtypes.fp16
    if dtype == thor.DataType.fp8_e5m2:
        return numpy_dtypes.fp16
    return numpy_dtypes.from_thor(dtype)


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    elif dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-2)
    elif dtype == thor.DataType.bf16:
        np.testing.assert_allclose(got, expected, rtol=4e-2, atol=4e-2)
    elif dtype == thor.DataType.fp8_e4m3:
        np.testing.assert_allclose(got, expected, rtol=2.5e-1, atol=2.5e-1)
    elif dtype == thor.DataType.fp8_e5m2:
        np.testing.assert_allclose(got, expected, rtol=3.5e-1, atol=3.5e-1)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


def _copy_numpy_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu_placement = Placement(DeviceType.cpu, 0)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)

    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))

    host_tensor = PhysicalTensor(cpu_placement, desc)
    host_tensor.numpy()[...] = arr.astype(_numpy_storage_dtype(dtype), copy=False)

    gpu_tensor = PhysicalTensor(gpu_placement, desc)
    gpu_tensor.copy_from_async(host_tensor, stream)
    return gpu_tensor


def _copy_gpu_to_numpy(tensor: PhysicalTensor, stream: Stream) -> np.ndarray:
    cpu_placement = Placement(DeviceType.cpu, 0)
    host_tensor = PhysicalTensor(cpu_placement, tensor.get_descriptor())
    host_tensor.copy_from_async(tensor, stream)
    stream.synchronize()
    return host_tensor.numpy().copy()


def _run_expr(expr, inputs: dict[str, tuple[np.ndarray, thor.DataType]], gpu_num: int = 0, use_fast_math: bool = False):
    stream = Stream(gpu_num=gpu_num)

    gpu_inputs = {}
    for name, (arr, dtype) in inputs.items():
        gpu_inputs[name] = _copy_numpy_to_gpu(arr, dtype, stream, gpu_num=gpu_num)

    eq = ex.compile(expr, device_num=gpu_num, use_fast_math=use_fast_math)
    stamped = eq.stamp(gpu_inputs, stream)
    stamped.run()

    return _copy_gpu_to_numpy(stamped.output(), stream)


def _run_outputs(
        outs, inputs: dict[str, tuple[np.ndarray, thor.DataType]], gpu_num: int = 0, use_fast_math: bool = False):
    stream = Stream(gpu_num=gpu_num)

    gpu_inputs = {}
    for name, (arr, dtype) in inputs.items():
        gpu_inputs[name] = _copy_numpy_to_gpu(arr, dtype, stream, gpu_num=gpu_num)

    eq = ex.compile(outs, device_num=gpu_num, use_fast_math=use_fast_math)
    stamped = eq.stamp(gpu_inputs, stream)
    stamped.run()

    result = {}
    for name in eq.output_names():
        result[name] = _copy_gpu_to_numpy(stamped.output(name), stream)
    return result


def _materialize(arr: np.ndarray, dtype: thor.DataType) -> np.ndarray:
    return arr.astype(_numpy_storage_dtype(dtype))


@pytest.mark.cuda
def test_input_as_type_single_input_casts_value_before_pointwise():
    x = ex.input("x", as_type=thor.DataType.fp16)
    expr = (x + 1.0) * 2.0

    runtime_dtype = thor.DataType.fp32
    graph_dtype = thor.DataType.fp16

    x_np = np.linspace(-1.0, 1.0, num=17, dtype=np.float32).astype(_numpy_storage_dtype(runtime_dtype))

    x_graph = _materialize(x_np, graph_dtype)
    t1 = _materialize(x_graph.astype(_numpy_compute_dtype(graph_dtype)) + 1.0, graph_dtype)
    expected = _materialize(t1.astype(_numpy_compute_dtype(graph_dtype)) * 2.0, graph_dtype)

    got = _run_expr(expr, {
        "x": (x_np, runtime_dtype)
    })

    assert got.dtype == _numpy_storage_dtype(graph_dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, graph_dtype)


@pytest.mark.cuda
def test_input_as_type_promotes_against_other_tensor():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")
    expr = x + y

    x_runtime_dtype = thor.DataType.fp32
    y_runtime_dtype = thor.DataType.fp32
    x_graph_dtype = thor.DataType.fp16
    out_dtype = thor.DataType.fp32

    x_np = np.linspace(-1.0, 1.0, num=15, dtype=np.float32).reshape(3, 5).astype(_numpy_storage_dtype(x_runtime_dtype))
    y_np = np.linspace(0.25, 2.25, num=15, dtype=np.float32).reshape(3, 5).astype(_numpy_storage_dtype(y_runtime_dtype))

    x_graph = _materialize(x_np, x_graph_dtype)
    expected = _materialize(
        x_graph.astype(_numpy_compute_dtype(out_dtype)) + y_np.astype(_numpy_compute_dtype(out_dtype)),
        out_dtype,
    )

    got = _run_expr(
        expr,
        {
            "x": (x_np, x_runtime_dtype),
            "y": (y_np, y_runtime_dtype),
        },
    )

    assert got.dtype == _numpy_storage_dtype(out_dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, out_dtype)


@pytest.mark.cuda
def test_input_as_type_direct_reduction_materializes_then_reduces():
    x = ex.input("x", as_type=thor.DataType.fp16)
    expr = ex.reduce_sum(x, axis=2, squeeze=False)

    runtime_dtype = thor.DataType.fp32
    graph_dtype = thor.DataType.fp16

    x_np = np.linspace(
        0.25, 2.25, num=24, dtype=np.float32).reshape(2, 3, 4).astype(_numpy_storage_dtype(runtime_dtype))

    x_graph = _materialize(x_np, graph_dtype)
    expected = np.sum(
        x_graph.astype(_numpy_compute_dtype(thor.DataType.fp32)),
        axis=2,
        keepdims=True,
    ).astype(_numpy_storage_dtype(graph_dtype))

    got = _run_expr(expr, {
        "x": (x_np, runtime_dtype)
    })

    assert got.dtype == _numpy_storage_dtype(graph_dtype)
    assert got.shape == (2, 3, 1)
    _assert_close(got, expected, graph_dtype)


@pytest.mark.cuda
def test_input_as_type_broadcast_mixed_outputs_and_shapes():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")

    xy = x + y
    y_shift = y - 0.5

    outs = ex.outputs(
        {
            "wide_sum": xy,  # [17, 3], fp32
            "wide_mix": xy * y_shift,  # [17, 3], fp32
            "narrow_shift": x + 1.0,  # [17, 1], fp16
        })

    x_runtime_dtype = thor.DataType.fp32
    y_runtime_dtype = thor.DataType.fp32

    x_np = np.linspace(
        0.25, 2.25, num=17, dtype=np.float32).reshape(17, 1).astype(_numpy_storage_dtype(x_runtime_dtype))
    y_np = np.linspace(1.0, 1.5, num=3, dtype=np.float32).reshape(1, 3).astype(_numpy_storage_dtype(y_runtime_dtype))

    x_graph = _materialize(x_np, thor.DataType.fp16)

    wide_sum = _materialize(
        x_graph.astype(_numpy_compute_dtype(thor.DataType.fp32)) +
        y_np.astype(_numpy_compute_dtype(thor.DataType.fp32)),
        thor.DataType.fp32,
    )

    y_shift_ref = _materialize(
        y_np.astype(_numpy_compute_dtype(thor.DataType.fp32)) - 0.5,
        thor.DataType.fp32,
    )

    wide_mix = _materialize(
        wide_sum.astype(_numpy_compute_dtype(thor.DataType.fp32)) *
        y_shift_ref.astype(_numpy_compute_dtype(thor.DataType.fp32)),
        thor.DataType.fp32,
    )

    narrow_shift = _materialize(
        x_graph.astype(_numpy_compute_dtype(thor.DataType.fp16)) + 1.0,
        thor.DataType.fp16,
    )

    got = _run_outputs(
        outs,
        {
            "x": (x_np, x_runtime_dtype),
            "y": (y_np, y_runtime_dtype),
        },
    )

    assert got["wide_sum"].dtype == _numpy_storage_dtype(thor.DataType.fp32)
    assert got["wide_mix"].dtype == _numpy_storage_dtype(thor.DataType.fp32)
    assert got["narrow_shift"].dtype == _numpy_storage_dtype(thor.DataType.fp16)

    assert got["wide_sum"].shape == (17, 3)
    assert got["wide_mix"].shape == (17, 3)
    assert got["narrow_shift"].shape == (17, 1)

    _assert_close(got["wide_sum"], wide_sum, thor.DataType.fp32)
    _assert_close(got["wide_mix"], wide_mix, thor.DataType.fp32)
    _assert_close(got["narrow_shift"], narrow_shift, thor.DataType.fp16)


@pytest.mark.cuda
def test_input_as_type_homogeneous_broadcast_noop_cast_same_runtime_dtype():
    x = ex.input("x", as_type=thor.DataType.fp16)
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    dtype = thor.DataType.fp16

    x_np = np.linspace(0.25, 2.25, num=17, dtype=np.float32).reshape(17, 1).astype(_numpy_storage_dtype(dtype))
    y_np = np.linspace(1.0, 1.5, num=3, dtype=np.float32).reshape(1, 3).astype(_numpy_storage_dtype(dtype))

    x_graph = _materialize(x_np, dtype)
    y_graph = _materialize(y_np, dtype)

    expected_sum = _materialize(
        x_graph.astype(_numpy_compute_dtype(dtype)) + y_graph.astype(_numpy_compute_dtype(dtype)),
        dtype,
    )
    expected_prod = _materialize(
        x_graph.astype(_numpy_compute_dtype(dtype)) * y_graph.astype(_numpy_compute_dtype(dtype)),
        dtype,
    )

    got = _run_outputs(
        outs,
        {
            "x": (x_np, dtype),
            "y": (y_np, dtype),
        },
    )

    assert got["sum"].dtype == _numpy_storage_dtype(dtype)
    assert got["prod"].dtype == _numpy_storage_dtype(dtype)

    assert got["sum"].shape == (17, 3)
    assert got["prod"].shape == (17, 3)

    _assert_close(got["sum"], expected_sum, dtype)
    _assert_close(got["prod"], expected_prod, dtype)
