import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

BACKWARD_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.fp32,
]

TRANSPOSE_CASES = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]


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


def _matmul_operand_arrays(transpose_a: bool, transpose_b: bool,
                           dtype: thor.DataType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = 2
    k = 3
    n = 4

    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)

    a_np = np.arange(1, 1 + np.prod(a_shape), dtype=np.float32).reshape(a_shape) / 4.0 - 1.0
    b_np = np.arange(1, 1 + np.prod(b_shape), dtype=np.float32).reshape(b_shape) / 5.0 - 0.75
    grad_np = np.arange(1, 1 + (m * n), dtype=np.float32).reshape((m, n)) / 7.0 - 0.5

    storage_dtype = _numpy_storage_dtype(dtype)
    return a_np.astype(storage_dtype), b_np.astype(storage_dtype), grad_np.astype(storage_dtype)


def _matmul_backward_reference(
        a: np.ndarray,
        b: np.ndarray,
        grad: np.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    a_ref = a.astype(np.float32)
    b_ref = b.astype(np.float32)
    grad_ref = grad.astype(np.float32)

    a_eff = a_ref.T if transpose_a else a_ref
    b_eff = b_ref.T if transpose_b else b_ref

    da_eff = alpha * (grad_ref @ b_eff.T)
    db_eff = alpha * (a_eff.T @ grad_ref)

    da = da_eff.T if transpose_a else da_eff
    db = db_eff.T if transpose_b else db_eff
    return da, db


def _gemm_backward_reference(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        grad: np.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    da, db = _matmul_backward_reference(a, b, grad, transpose_a=transpose_a, transpose_b=transpose_b, alpha=alpha)
    dc = beta * grad.astype(np.float32)
    return da, db, dc


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("transpose_a,transpose_b", TRANSPOSE_CASES)
def test_matmul_backward_all_transpose_combinations_numerical(
        dtype: thor.DataType, transpose_a: bool, transpose_b: bool):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad"]

    a_np, b_np, grad_np = _matmul_operand_arrays(transpose_a, transpose_b, dtype)
    expected_a_grad, expected_b_grad = _matmul_backward_reference(
        a_np,
        b_np,
        grad_np,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_matmul_backward_pointwise_composition_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    out = ex.exp(ex.matmul(a, b, transpose_a=True, transpose_b=False) * 0.25 + 0.5)
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)

    a_np, b_np, grad_np = _matmul_operand_arrays(True, False, dtype)

    mm = a_np.astype(np.float32).T @ b_np.astype(np.float32)
    local_grad = grad_np.astype(np.float32) * np.exp(mm * 0.25 + 0.5) * 0.25
    expected_a_grad, expected_b_grad = _matmul_backward_reference(
        a_np,
        b_np,
        local_grad,
        transpose_a=True,
        transpose_b=False,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_matmul_backward_requested_subset_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.matmul(a, b, transpose_a=False, transpose_b=True), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad"]

    a_np, b_np, grad_np = _matmul_operand_arrays(False, True, dtype)
    expected_a_grad, _ = _matmul_backward_reference(a_np, b_np, grad_np, transpose_a=False, transpose_b=True)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_matmul_backward_accumulate_grad_outputs_stamp_uses_provided_accumulators_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.matmul(a, b), device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["a", "b"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    a_np, b_np, grad_np = _matmul_operand_arrays(False, False, dtype)
    computed_a_grad, computed_b_grad = _matmul_backward_reference(a_np, b_np, grad_np)

    prefill_a_grad_np = np.array([[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]],
                                 dtype=np.float32).astype(_numpy_storage_dtype(dtype))
    prefill_b_grad_np = np.array(
        [[-1.0, -2.0, -3.0, -4.0], [1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 0.25, -0.25]],
        dtype=np.float32).astype(_numpy_storage_dtype(dtype))

    expected = {
        "a_grad": _cast_reference_to_storage_dtype(prefill_a_grad_np.astype(np.float32) + computed_a_grad, dtype),
        "b_grad": _cast_reference_to_storage_dtype(prefill_b_grad_np.astype(np.float32) + computed_b_grad, dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }
    outputs_gpu = {
        "a_grad": _host_to_gpu(prefill_a_grad_np, dtype, stream),
        "b_grad": _host_to_gpu(prefill_b_grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream=stream, preallocated_outputs=outputs_gpu)
    stamped.run()

    got_a_grad = _copy_to_host(outputs_gpu["a_grad"], dtype, stream)
    got_b_grad = _copy_to_host(outputs_gpu["b_grad"], dtype, stream)

    _assert_close(got_a_grad, expected["a_grad"], dtype)
    _assert_close(got_b_grad, expected["b_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_gemm_backward_alpha_beta_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"
    alpha = 1.25
    beta = -0.5

    fwd_eq = ex.compile(
        ex.gemm(a, b, c, alpha=alpha, beta=beta, transpose_a=True, transpose_b=False),
        device_num=0,
    )
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad", "c_grad"]

    a_np, b_np, grad_np = _matmul_operand_arrays(True, False, dtype)
    c_np = np.array(
        [[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(dtype))

    expected_a_grad, expected_b_grad, expected_c_grad = _gemm_backward_reference(
        a_np,
        b_np,
        c_np,
        grad_np,
        transpose_a=True,
        transpose_b=False,
        alpha=alpha,
        beta=beta,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)
    got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)
    _assert_close(got_c_grad, _cast_reference_to_storage_dtype(expected_c_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_gemm_backward_beta_zero_aux_grad_is_zero(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.gemm(a, b, c, alpha=0.75, beta=0.0), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["c"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["c_grad"]

    a_np, b_np, grad_np = _matmul_operand_arrays(False, False, dtype)
    c_np = np.array(
        [[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(dtype))

    expected_c_grad = np.zeros_like(c_np.astype(np.float32))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)
    _assert_close(got_c_grad, _cast_reference_to_storage_dtype(expected_c_grad, dtype), dtype)


@pytest.mark.cuda
def test_gemm_backward_rejects_transpose_c():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")

    fwd_eq = ex.compile(ex.gemm(a, b, c, transpose_c=True), device_num=0)

    with pytest.raises(RuntimeError, match="transpose_aux|transposeC|transpose_c"):
        fwd_eq.compile_backward(["c"], error_input_name="__grad_output")


@pytest.mark.cuda
def test_matmul_backward_large_fp16_likely_workspace_numerical():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"

    # Chosen to be large enough to make workspace-backed matmul algorithms likely,
    # while still being reasonable for a unit test.
    m = 512
    k = 768
    n = 640

    a = ex.input("a")
    b = ex.input("b")

    # Add a little pointwise structure so backward is not just a bare matmul.
    out = ex.exp((a @ b) * 0.125 - 0.25)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad"]

    # Keep values small so the exp stays well behaved in fp16.
    a_ref = (((np.arange(m * k, dtype=np.float32).reshape(m, k) % 23.0) - 11.0) / 64.0)
    b_ref = (((np.arange(k * n, dtype=np.float32).reshape(k, n) % 19.0) - 9.0) / 64.0)
    upstream_ref = (((np.arange(m * n, dtype=np.float32).reshape(m, n) % 17.0) - 8.0) / 32.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = a_ref.astype(storage_dtype)
    b_np = b_ref.astype(storage_dtype)
    grad_np = upstream_ref.astype(storage_dtype)

    mm = a_ref @ b_ref
    local_grad = upstream_ref * np.exp(mm * 0.125 - 0.25) * 0.125

    expected_a_grad = local_grad @ b_ref.T
    expected_b_grad = a_ref.T @ local_grad

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)


@pytest.mark.cuda
def test_operator_gemm_pattern_right_scaled_matmul_and_addend_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")

    alpha = 0.75
    beta = 1.25

    expr = (a @ b) * alpha + c * beta
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, -1.0, 2.0]], dtype=np.float32)
    b_np = np.array([[0.25, 1.5], [-0.5, 2.0], [1.25, -1.0]], dtype=np.float32)
    c_np = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)

    expected = (a_np @ b_np) * alpha + c_np * beta

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
