import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

TILE_ALIGNED_SHAPE = (64, 96)
RAGGED_SHAPE = (35, 67)

SAME_STORAGE_WIDTH_DTYPE_CASES = [
    pytest.param(thor.DataType.fp32, thor.DataType.fp32, id="fp32_to_fp32"),
    pytest.param(thor.DataType.fp16, thor.DataType.fp16, id="fp16_to_fp16"),
    pytest.param(thor.DataType.fp16, thor.DataType.bf16, id="fp16_to_bf16"),
    pytest.param(thor.DataType.bf16, thor.DataType.fp16, id="bf16_to_fp16"),
    pytest.param(thor.DataType.bf16, thor.DataType.bf16, id="bf16_to_bf16"),
    pytest.param(thor.DataType.fp8_e4m3, thor.DataType.fp8_e4m3, id="fp8_e4m3_to_fp8_e4m3"),
    pytest.param(thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2, id="fp8_e4m3_to_fp8_e5m2"),
    pytest.param(thor.DataType.fp8_e5m2, thor.DataType.fp8_e4m3, id="fp8_e5m2_to_fp8_e4m3"),
    pytest.param(thor.DataType.fp8_e5m2, thor.DataType.fp8_e5m2, id="fp8_e5m2_to_fp8_e5m2"),
]

CROSS_STORAGE_WIDTH_DTYPE_CASES = [
    pytest.param(thor.DataType.fp32, thor.DataType.fp16, id="fp32_to_fp16"),
    pytest.param(thor.DataType.fp32, thor.DataType.bf16, id="fp32_to_bf16"),
    pytest.param(thor.DataType.fp32, thor.DataType.fp8_e4m3, id="fp32_to_fp8_e4m3"),
    pytest.param(thor.DataType.fp32, thor.DataType.fp8_e5m2, id="fp32_to_fp8_e5m2"),
    pytest.param(thor.DataType.fp16, thor.DataType.fp32, id="fp16_to_fp32"),
    pytest.param(thor.DataType.bf16, thor.DataType.fp32, id="bf16_to_fp32"),
    pytest.param(thor.DataType.fp16, thor.DataType.fp8_e4m3, id="fp16_to_fp8_e4m3"),
    pytest.param(thor.DataType.bf16, thor.DataType.fp8_e5m2, id="bf16_to_fp8_e5m2"),
    pytest.param(thor.DataType.fp8_e4m3, thor.DataType.fp16, id="fp8_e4m3_to_fp16"),
    pytest.param(thor.DataType.fp8_e5m2, thor.DataType.bf16, id="fp8_e5m2_to_bf16"),
    pytest.param(thor.DataType.fp8_e4m3, thor.DataType.fp32, id="fp8_e4m3_to_fp32"),
    pytest.param(thor.DataType.fp8_e5m2, thor.DataType.fp32, id="fp8_e5m2_to_fp32"),
]


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _numpy_compute_dtype(dtype: thor.DataType) -> np.dtype:
    if dtype in (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2):
        return numpy_dtypes.fp16
    return _numpy_storage_dtype(dtype)


def _default_compute_dtype(input_dtype: thor.DataType, output_dtype: thor.DataType) -> thor.DataType:
    if input_dtype == thor.DataType.fp32 or output_dtype == thor.DataType.fp32:
        return thor.DataType.fp32
    if {input_dtype, output_dtype} == {thor.DataType.fp16, thor.DataType.bf16}:
        return thor.DataType.fp32
    if input_dtype in (thor.DataType.fp16, thor.DataType.bf16):
        return input_dtype
    if output_dtype in (thor.DataType.fp16, thor.DataType.bf16):
        return output_dtype
    if input_dtype in (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2) and output_dtype in (thor.DataType.fp8_e4m3,
                                                                                            thor.DataType.fp8_e5m2):
        return thor.DataType.fp16
    raise AssertionError(f"Unhandled dtype pair: {input_dtype}, {output_dtype}")


def _cast_reference_to_storage_dtype(values: np.ndarray, dtype: thor.DataType) -> np.ndarray:
    values32 = values.astype(np.float32)
    if dtype == thor.DataType.fp8_e4m3:
        values32 = np.clip(values32, -448.0, 448.0)
    return values32.astype(_numpy_storage_dtype(dtype))


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)

    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got32, expected32, rtol=2e-5, atol=2e-6)
    elif dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got32, expected32, rtol=4e-2, atol=4e-2)
    elif dtype == thor.DataType.bf16:
        np.testing.assert_allclose(got32, expected32, rtol=5e-2, atol=5e-2)
    elif dtype == thor.DataType.fp8_e4m3:
        np.testing.assert_allclose(got32, expected32, rtol=2.5e-1, atol=2.5e-1)
    elif dtype == thor.DataType.fp8_e5m2:
        np.testing.assert_allclose(got32, expected32, rtol=3.5e-1, atol=5e-1)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


def _copy_numpy_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))

    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr.astype(_numpy_storage_dtype(dtype), copy=False)

    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _copy_gpu_to_numpy(tensor: PhysicalTensor, stream: Stream) -> np.ndarray:
    cpu = Placement(DeviceType.cpu, 0)
    host = PhysicalTensor(cpu, tensor.get_descriptor())
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


def _make_matrix(shape: tuple[int, int], dtype: thor.DataType, *, offset: float) -> np.ndarray:
    numel = shape[0] * shape[1]
    values = np.linspace(-1.0 + offset, 1.0 + offset, num=numel, dtype=np.float32).reshape(shape)
    return _cast_reference_to_storage_dtype(values, dtype)


def _arithmetic_reference(x_np: np.ndarray, y_np: np.ndarray, compute_dtype: thor.DataType) -> np.ndarray:
    x = x_np.astype(_numpy_compute_dtype(compute_dtype))
    y = y_np.astype(_numpy_compute_dtype(compute_dtype))
    return (x * 1.25) + (y * -0.5) + ((x - y) * 0.125) - 0.25


def _assert_fused_transpose_stage(eq, inputs_gpu: dict[str, PhysicalTensor]):
    # A folded transposed materialization is still a fused stage at the execution-plan level.
    # A separate staged transpose would show up as an additional "Transpose" stage here.
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]


def _run_fused_transpose_expr(
        input_dtype: thor.DataType,
        output_dtype: thor.DataType,
        shape: tuple[int, int],
        *,
        forced_compute_dtype: thor.DataType | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = ex.input("x")
    y = ex.input("y")

    fused = (x * 1.25) + (y * -0.5) + ((x - y) * 0.125) - 0.25
    if forced_compute_dtype is None:
        materialized = fused.with_output_dtype(output_dtype)
        reference_compute_dtype = _default_compute_dtype(input_dtype, output_dtype)
    else:
        materialized = fused.with_dtypes(output_dtype=output_dtype, compute_dtype=forced_compute_dtype)
        reference_compute_dtype = forced_compute_dtype
    expr = materialized.transpose()

    x_np = _make_matrix(shape, input_dtype, offset=-0.125)
    y_np = _make_matrix(shape, input_dtype, offset=0.375)

    expected_untransposed = _cast_reference_to_storage_dtype(
        _arithmetic_reference(x_np, y_np, reference_compute_dtype),
        output_dtype,
    )
    expected = expected_untransposed.T

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _copy_numpy_to_gpu(x_np, input_dtype, stream),
        "y": _copy_numpy_to_gpu(y_np, input_dtype, stream),
    }

    eq = ex.compile(expr, device_num=0)
    _assert_fused_transpose_stage(eq, inputs_gpu)

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_gpu_to_numpy(stamped.output(), stream)
    return got, expected


@pytest.mark.cuda
@pytest.mark.parametrize("input_dtype,output_dtype", SAME_STORAGE_WIDTH_DTYPE_CASES)
@pytest.mark.parametrize("shape", [TILE_ALIGNED_SHAPE, RAGGED_SHAPE], ids=["tile_aligned", "ragged_edges"])
def test_fused_arithmetic_transpose_same_storage_width_numerical(
        input_dtype: thor.DataType, output_dtype: thor.DataType, shape: tuple[int, int]):
    got, expected = _run_fused_transpose_expr(input_dtype, output_dtype, shape)

    assert got.dtype == _numpy_storage_dtype(output_dtype)
    assert got.shape == expected.shape == (shape[1], shape[0])
    _assert_close(got, expected, output_dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("input_dtype,output_dtype", CROSS_STORAGE_WIDTH_DTYPE_CASES)
@pytest.mark.parametrize("shape", [TILE_ALIGNED_SHAPE, RAGGED_SHAPE], ids=["tile_aligned", "ragged_edges"])
def test_fused_arithmetic_transpose_cross_storage_width_conversion_numerical(
        input_dtype: thor.DataType, output_dtype: thor.DataType, shape: tuple[int, int]):
    got, expected = _run_fused_transpose_expr(input_dtype, output_dtype, shape)

    assert got.dtype == _numpy_storage_dtype(output_dtype)
    assert got.shape == expected.shape == (shape[1], shape[0])
    _assert_close(got, expected, output_dtype)
