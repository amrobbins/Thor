import subprocess

import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

MATMUL_DTYPES = [
    thor.DataType.fp16,
    # thor.DataType.bf16,
    thor.DataType.fp32,
]


def _gpu_compute_capability(gpu_num: int = 0) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_num}",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=True,
        )
    except Exception:
        return None

    lines = (result.stdout or "").strip().splitlines()
    if not lines:
        return None
    line = lines[0].strip()
    if not line:
        return None

    try:
        major_s, minor_s = line.split(".", 1)
        return int(major_s), int(minor_s)
    except Exception:
        return None


def _gpu_supports_bf16_matmul(gpu_num: int = 0) -> bool:
    capability = _gpu_compute_capability(gpu_num)
    if capability is None:
        return False
    major, _minor = capability
    return major >= 8


def _gpu_supports_fp8_matmul(gpu_num: int = 0) -> bool:
    capability = _gpu_compute_capability(gpu_num)
    if capability is None:
        return False
    major, minor = capability
    return major >= 9 or (major == 8 and minor >= 9)


def _assert_allclose_fp32_output(got: np.ndarray, expected: np.ndarray, rtol: float, atol: float):
    assert got.dtype == np.float32
    np.testing.assert_allclose(got.astype(np.float32), expected.astype(np.float32), rtol=rtol, atol=atol)


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _cast_reference_to_storage_dtype(values: np.ndarray, dtype: thor.DataType) -> np.ndarray:
    return values.astype(np.float32).astype(_numpy_storage_dtype(dtype))


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)

    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got32, expected32, rtol=1e-4, atol=1e-5)
    elif dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got32, expected32, rtol=5e-2, atol=5e-2)
    elif dtype == thor.DataType.bf16:
        np.testing.assert_allclose(got32, expected32, rtol=7e-2, atol=7e-2)
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
def test_matmul_bf16_inputs_fp32_output_numerical():
    if not _gpu_supports_bf16_matmul(0):
        pytest.skip("BF16 GEMM kernels require an Ampere-or-newer GPU.")

    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(ex.matmul(a, b, output_dtype=thor.DataType.fp32), device_num=0)

    a_dtype = thor.DataType.bf16
    b_dtype = thor.DataType.bf16
    output_dtype = thor.DataType.fp32

    a_np = np.array(
        [[0.25, -0.5, 1.0, -1.25], [1.5, 0.75, -0.25, 0.5], [-1.0, 1.25, 0.375, -0.75]],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(a_dtype))
    b_np = np.array(
        [
            [1.0, -0.25, 0.5, 1.25, -1.5], [-0.75, 0.5, -1.0, 0.25, 1.0], [0.375, -1.25, 0.75, -0.5, 0.625],
            [1.5, 0.25, -0.375, 1.0, -0.875]
        ],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(b_dtype))

    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, a_dtype, stream),
        "b": _host_to_gpu(b_np, b_dtype, stream),
    }

    assert eq._debug_stage_kinds(inputs_gpu) == ["Matmul(lhsT=0,rhsT=0,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), output_dtype, stream)
    _assert_allclose_fp32_output(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda
def test_gemm_fp16_inputs_fp32_addend_and_output_numerical():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(ex.gemm(a, b, c, alpha=0.75, beta=-1.25, output_dtype=thor.DataType.fp32), device_num=0)

    ab_dtype = thor.DataType.fp16
    c_dtype = thor.DataType.fp32
    output_dtype = thor.DataType.fp32

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(_numpy_storage_dtype(ab_dtype))
    b_np = np.array(
        [[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(ab_dtype))
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    expected = (a_np.astype(np.float32) @ b_np.astype(np.float32)) * 0.75 + c_np * -1.25

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, ab_dtype, stream),
        "b": _host_to_gpu(b_np, ab_dtype, stream),
        "c": _host_to_gpu(c_np, c_dtype, stream),
    }

    assert eq._debug_stage_kinds(inputs_gpu) == ["Matmul(lhsT=0,rhsT=0,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), output_dtype, stream)
    _assert_allclose_fp32_output(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda
def test_operator_gemm_pattern_preserves_fp16_inputs_fp32_addend_and_output_numerical():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile((a @ b) * 0.5 + c * -0.25, device_num=0)

    ab_dtype = thor.DataType.fp16
    c_dtype = thor.DataType.fp32
    output_dtype = thor.DataType.fp32

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(_numpy_storage_dtype(ab_dtype))
    b_np = np.array(
        [[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(ab_dtype))
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    expected = (a_np.astype(np.float32) @ b_np.astype(np.float32)) * 0.5 + c_np * -0.25

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, ab_dtype, stream),
        "b": _host_to_gpu(b_np, ab_dtype, stream),
        "c": _host_to_gpu(c_np, c_dtype, stream),
    }

    # This guards the GEMM pattern path, not just the explicit ex.gemm API.
    assert eq._debug_stage_kinds(inputs_gpu) == ["Matmul(lhsT=0,rhsT=0,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), output_dtype, stream)
    _assert_allclose_fp32_output(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_matmul_dunder_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(a @ b, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(storage_dtype)

    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_matmul_transpose_followed_by_pointwise_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    expr = ex.exp(ex.matmul(a, b, transpose_a=True, transpose_b=True) * 0.25 + 0.5)
    eq = ex.compile(expr, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[1.5, -0.5, 2.0], [0.25, 1.0, -1.0], [2.5, -2.0, 0.75], [-1.5, 0.5, 1.25]],
                    dtype=np.float32).astype(storage_dtype)

    mm = a_np.astype(np.float32).T @ b_np.astype(np.float32).T
    expected = np.exp(mm * 0.25 + 0.5)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_gemm_alpha_beta_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 1.25
    beta = -0.5
    eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(storage_dtype)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32).astype(storage_dtype)

    expected = alpha * (a_np.astype(np.float32) @ b_np.astype(np.float32)) + beta * c_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_operator_gemm_runtime_scalars_right_scaled_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")

    expr = (a @ b) * alpha + c * beta
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)

    alpha_v1 = 0.75
    beta_v1 = -1.25
    stamped.run({
        "alpha": alpha_v1,
        "beta": beta_v1
    })
    got_v1 = _copy_to_host(stamped.output(), dtype, stream)
    expected_v1 = (a_np @ b_np) * alpha_v1 + c_np * beta_v1
    _assert_close(got_v1, expected_v1, dtype)

    alpha_v2 = -0.5
    beta_v2 = 0.25
    stamped.run({
        "alpha": alpha_v2,
        "beta": beta_v2
    })
    got_v2 = _copy_to_host(stamped.output(), dtype, stream)
    expected_v2 = (a_np @ b_np) * alpha_v2 + c_np * beta_v2
    _assert_close(got_v2, expected_v2, dtype)


@pytest.mark.cuda
def test_gemm_transpose_ab_and_preallocated_output_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 0.75
    beta = 1.5

    expr = ex.gemm(
        a,
        b,
        c,
        alpha=alpha,
        beta=beta,
        transpose_a=True,
        transpose_b=True,
    )
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0]], dtype=np.float32)  # 3x2
    b_np = np.array(
        [[1.5, -0.5, 2.0], [0.25, 1.0, -1.0], [2.5, -2.0, 0.75], [-1.5, 0.5, 1.25]], dtype=np.float32)  # 4x3
    c_np = np.array([[0.2, -1.0, 0.5, 2.0], [-0.75, 0.25, 1.5, -0.5]], dtype=np.float32)  # 2x4

    expected = alpha * (a_np.T @ b_np.T) + beta * c_np

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }
    preallocated_output = _gpu_tensor([2, 4], dtype)

    stamped = eq.stamp(inputs_gpu, stream, preallocated_output=preallocated_output)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_matmul_output_dtype_override_and_preallocated_output_numerical():
    input_dtype = thor.DataType.fp16
    output_dtype = thor.DataType.fp16

    a = ex.input("a")
    b = ex.input("b")
    expr = ex.matmul(a, b, compute_dtype=thor.DataType.fp32, output_dtype=output_dtype)
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(_numpy_storage_dtype(input_dtype))
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(_numpy_storage_dtype(input_dtype))
    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, input_dtype, stream),
        "b": _host_to_gpu(b_np, input_dtype, stream),
    }
    preallocated_output = _gpu_tensor([2, 4], output_dtype)

    stamped = eq.stamp(inputs_gpu, stream, preallocated_output=preallocated_output)
    stamped.run()

    got = _copy_to_host(stamped.output(), output_dtype, stream)
    _assert_close(got, expected, output_dtype)


@pytest.mark.cuda
def test_matmul_rejects_incompatible_dimensions_at_stamp_time():
    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(ex.matmul(a, b), device_num=0)

    dtype = thor.DataType.fp32
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(np.ones((2, 3), dtype=np.float32), dtype, stream),
        "b": _host_to_gpu(np.ones((5, 4), dtype=np.float32), dtype, stream),
    }

    with pytest.raises(RuntimeError, match="incompatible matrix dimensions"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_gemm_rejects_incompatible_addend_dimensions_at_stamp_time():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(ex.gemm(a, b, c), device_num=0)

    dtype = thor.DataType.fp32
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(np.ones((2, 3), dtype=np.float32), dtype, stream),
        "b": _host_to_gpu(np.ones((3, 4), dtype=np.float32), dtype, stream),
        "c": _host_to_gpu(np.ones((4, 2), dtype=np.float32), dtype, stream),
    }

    with pytest.raises(RuntimeError, match="addend|dimensions|incompatible"):
        eq.stamp(inputs_gpu, stream)


def test_matmul_rdunder():
    a = ex.input("a")
    b = ex.input("b")
    expr1 = a.__rmatmul__(b)
    expr2 = ex.matmul(b, a)
    eq1 = ex.compile(expr1, device_num=0)
    eq2 = ex.compile(expr2, device_num=0)
    assert eq1 is not None
    assert eq2 is not None


def test_matmul_idunder():
    a = ex.input("a")
    b = ex.input("b")
    c = a.__imatmul__(b)
    eq = ex.compile(c, device_num=0)
    assert eq is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_operator_gemm_pattern_matmul_plus_addend_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(a @ b + c, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(storage_dtype)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32).astype(storage_dtype)

    expected = a_np.astype(np.float32) @ b_np.astype(np.float32) + c_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_operator_gemm_pattern_scaled_and_reordered_numerical():
    dtype = thor.DataType.fp32
    alpha = 0.75
    beta = -1.25

    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(beta * c + alpha * (a @ b), device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    expected = beta * c_np + alpha * (a_np @ b_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_operator_matmul_plus_scalar_stays_valid_and_is_not_over_lowered():
    dtype = thor.DataType.fp32

    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(a @ b + 1.5, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    expected = (a_np @ b_np) + 1.5

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_tensor_runtime_scalars_explicit_api_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha")
    beta = ex.tensor_runtime_scalar("beta")

    expr = ex.gemm(a, b, c, alpha=alpha, beta=beta)
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _host_to_gpu(np.array([0.75], dtype=np.float32), dtype, stream)
    beta_buffer = _host_to_gpu(np.array([-1.25], dtype=np.float32), dtype, stream)
    tensor_scalar_inputs = {
        "alpha": thor.physical.TensorScalarBinding(alpha_buffer, 0, dtype),
        "beta": thor.physical.TensorScalarBinding(beta_buffer, 0, dtype),
    }

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    expected = (a_np @ b_np) * 0.75 + c_np * -1.25
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_operator_gemm_tensor_runtime_scalars_right_scaled_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha")
    beta = ex.tensor_runtime_scalar("beta")

    expr = (a @ b) * alpha + c * beta
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _host_to_gpu(np.array([-0.5], dtype=np.float32), dtype, stream)
    beta_buffer = _host_to_gpu(np.array([0.25], dtype=np.float32), dtype, stream)
    tensor_scalar_inputs = {
        "alpha": thor.physical.TensorScalarBinding(alpha_buffer, 0, dtype),
        "beta": thor.physical.TensorScalarBinding(beta_buffer, 0, dtype),
    }

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    expected = (a_np @ b_np) * -0.5 + c_np * 0.25
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_tensor_runtime_scalar_alpha_constant_beta_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha")

    expr = ex.gemm(a, b, c, alpha=alpha, beta=0.25)
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _host_to_gpu(np.array([0.75], dtype=np.float32), dtype, stream)
    tensor_scalar_inputs = {
        "alpha": thor.physical.TensorScalarBinding(alpha_buffer, 0, dtype),
    }

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    expected = (a_np @ b_np) * 0.75 + c_np * 0.25
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_tensor_runtime_scalars_with_extra_host_scaling_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha") * 0.5
    beta = ex.tensor_runtime_scalar("beta") * -2.0

    expr = ex.gemm(a, b, c, alpha=alpha, beta=beta)
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]], dtype=np.float32)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _host_to_gpu(np.array([1.5], dtype=np.float32), dtype, stream)
    beta_buffer = _host_to_gpu(np.array([-0.125], dtype=np.float32), dtype, stream)
    tensor_scalar_inputs = {
        "alpha": thor.physical.TensorScalarBinding(alpha_buffer, 0, dtype),
        "beta": thor.physical.TensorScalarBinding(beta_buffer, 0, dtype),
    }

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    expected = (a_np @ b_np) * (1.5 * 0.5) + c_np * (-0.125 * -2.0)
    _assert_close(got, expected, dtype)
