import math

import numpy as np
import pytest
import thor
from thor.physical import (
    AttentionMaskKind,
    AttentionTensorLayout,
    DeviceType,
    cudnn_frontend_attention_available,
    Expression as ex,
    PhysicalTensor,
    Placement,
    Stream,
    numpy_dtypes,
)

ATTENTION_DTYPES = [thor.DataType.fp16, thor.DataType.bf16]


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


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    host = _cpu_tensor(list(arr.shape), dtype)
    host.numpy()[...] = arr.astype(_numpy_storage_dtype(dtype))
    device = _gpu_tensor(list(arr.shape), dtype, gpu_num=gpu_num)
    device.copy_from_async(host, stream)
    return device


def _copy_to_host(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


def _cast_reference_to_storage_dtype(values: np.ndarray, dtype: thor.DataType) -> np.ndarray:
    return values.astype(np.float32).astype(_numpy_storage_dtype(dtype))


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)
    if dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got32, expected32, rtol=5e-2, atol=5e-2)
    elif dtype == thor.DataType.bf16:
        np.testing.assert_allclose(got32, expected32, rtol=8e-2, atol=8e-2)
    else:
        raise AssertionError(f"Unhandled attention test dtype: {dtype}")


def _attention_inputs(
    *,
    batch: int = 2,
    query_heads: int = 4,
    kv_heads: int = 4,
    query_len: int = 5,
    kv_len: int = 5,
    qk_dim: int = 16,
    v_dim: int = 16,
    dtype: thor.DataType = thor.DataType.fp16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    storage_dtype = _numpy_storage_dtype(dtype)
    q = rng.normal(0.0, 0.35, size=(batch, query_heads, query_len, qk_dim)).astype(np.float32)
    k = rng.normal(0.0, 0.30, size=(batch, kv_heads, kv_len, qk_dim)).astype(np.float32)
    v = rng.normal(0.0, 0.40, size=(batch, kv_heads, kv_len, v_dim)).astype(np.float32)
    return q.astype(storage_dtype), k.astype(storage_dtype), v.astype(storage_dtype)


def _mask_for(kind: AttentionMaskKind, query_len: int, kv_len: int, left: int = 0, right: int = 0) -> np.ndarray | None:
    q_pos = np.arange(query_len)[:, None]
    kv_pos = np.arange(kv_len)[None, :]

    if kind == AttentionMaskKind.none:
        return None
    if kind == AttentionMaskKind.causal_top_left:
        return kv_pos <= q_pos
    if kind == AttentionMaskKind.sliding_window_top_left:
        return (kv_pos >= (q_pos - left)) & (kv_pos <= (q_pos + right))
    if kind == AttentionMaskKind.causal_bottom_right:
        offset = kv_len - query_len
        return kv_pos <= (q_pos + offset)
    if kind == AttentionMaskKind.sliding_window_bottom_right:
        offset = kv_len - query_len
        center = q_pos + offset
        return (kv_pos >= (center - left)) & (kv_pos <= (center + right))
    raise AssertionError(f"Unhandled mask kind: {kind}")


def _attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | None = None,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    diagonal_left_bound: int = 0,
    diagonal_right_bound: int = 0,
) -> np.ndarray:
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    v32 = v.astype(np.float32)

    batch, query_heads, query_len, qk_dim = q32.shape
    _, kv_heads, kv_len, _ = k32.shape
    if query_heads % kv_heads != 0:
        raise ValueError("query_heads must be an integer multiple of kv_heads")
    repeat = query_heads // kv_heads
    kv_head_for_query_head = np.arange(query_heads) // repeat
    k_expanded = k32[:, kv_head_for_query_head, :, :]
    v_expanded = v32[:, kv_head_for_query_head, :, :]

    effective_scale = scale if scale is not None else 1.0 / math.sqrt(float(qk_dim))
    scores = np.einsum("bhsd,bhtd->bhst", q32, k_expanded) * np.float32(effective_scale)

    mask = _mask_for(mask_kind, query_len, kv_len, diagonal_left_bound, diagonal_right_bound)
    if mask is not None:
        scores = np.where(mask[None, None, :, :], scores, np.float32(-1.0e30))

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores).astype(np.float32)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", probs, v_expanded).astype(np.float32)


def _compile_attention(
    *,
    dtype: thor.DataType,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    attention_scale: float | None = None,
    diagonal_left_bound: int = 0,
    diagonal_right_bound: int = 0,
    q_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd,
    k_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd,
    v_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd,
    o_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd,
):
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    return ex.compile(
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_layout=q_layout,
            k_layout=k_layout,
            v_layout=v_layout,
            o_layout=o_layout,
            mask_kind=mask_kind,
            diagonal_left_bound=diagonal_left_bound,
            diagonal_right_bound=diagonal_right_bound,
            attention_scale=attention_scale,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ),
        device_num=0,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test_attention_forward_mha_matches_reference(dtype: thor.DataType):
    scale = 0.75 / math.sqrt(16.0)
    eq = _compile_attention(dtype=dtype, attention_scale=scale)
    q_np, k_np, v_np = _attention_inputs(dtype=dtype)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4, 5, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "query_heads,kv_heads",
    [
        (4, 1),  # MQA
        (4, 2),  # GQA
    ],
)
def test_attention_forward_mqa_and_gqa_match_reference(query_heads: int, kv_heads: int):
    dtype = thor.DataType.fp16
    eq = _compile_attention(dtype=dtype)
    q_np, k_np, v_np = _attention_inputs(query_heads=query_heads, kv_heads=kv_heads, query_len=4, kv_len=6, dtype=dtype)
    expected = _attention_reference(q_np, k_np, v_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, query_heads, 4, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_causal_top_left_mask_matches_reference():
    dtype = thor.DataType.fp16
    eq = _compile_attention(dtype=dtype, mask_kind=AttentionMaskKind.causal_top_left)
    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=5, kv_len=5, dtype=dtype)
    expected = _attention_reference(q_np, k_np, v_np, mask_kind=AttentionMaskKind.causal_top_left)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_stage_feeds_fused_epilogue_and_multi_output_reuses_stage():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    attn = ex.scaled_dot_product_attention(q, k, v, output_dtype=dtype, compute_dtype=thor.DataType.fp32)
    shifted = (attn * 1.25 + 0.125).with_output_dtype(dtype)
    outputs = ex.outputs({
        "attention": attn,
        "shifted": shifted,
    })
    eq = ex.compile(outputs, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    expected_attention = _attention_reference(q_np, k_np, v_np)
    expected_shifted = expected_attention * np.float32(1.25) + np.float32(0.125)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shapes(inputs_gpu) == {
        "attention": [1, 2, 4, 16],
        "shifted": [1, 2, 4, 16]
    }
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention", "FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    got_attention = _copy_to_host(got["attention"], dtype, stream)
    got_shifted = _copy_to_host(got["shifted"], dtype, stream)
    _assert_close(got_attention, _cast_reference_to_storage_dtype(expected_attention, dtype), dtype)
    _assert_close(got_shifted, _cast_reference_to_storage_dtype(expected_shifted, dtype), dtype)


@pytest.mark.cuda
def test_attention_bshd_layout_options_stamp_without_teaching_planner_new_stage_types():
    dtype = thor.DataType.fp16
    eq = _compile_attention(
        dtype=dtype,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
    )
    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 2, 4, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    # Stamping validates the layout/options descriptor.  This deliberately does
    # not run numerically because the default Thor tensor fill path is row-major
    # BHSD; testing BSHD numerics needs a physical-layout-aware tensor fixture.
    eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_attention_invalid_gqa_head_ratio_raises_before_execution():
    dtype = thor.DataType.fp16
    eq = _compile_attention(dtype=dtype)
    q_np, k_np, v_np = _attention_inputs(query_heads=3, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    with pytest.raises(RuntimeError, match="integer multiple"):
        eq.output_shape(inputs_gpu)


@pytest.mark.cuda
def test_attention_invalid_qk_head_dimension_raises_during_shape_resolution():
    dtype = thor.DataType.fp16
    eq = _compile_attention(dtype=dtype)
    q_np = np.zeros((1, 2, 4, 16), dtype=_numpy_storage_dtype(dtype))
    k_np = np.zeros((1, 2, 4, 24), dtype=_numpy_storage_dtype(dtype))
    v_np = np.zeros((1, 2, 4, 16), dtype=_numpy_storage_dtype(dtype))
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    with pytest.raises(RuntimeError, match="q/k head dimensions"):
        eq.output_shape(inputs_gpu)


@pytest.mark.cuda
def test_attention_invalid_cudnn_head_dim_multiple_raises_at_stamp():
    dtype = thor.DataType.fp16
    eq = _compile_attention(dtype=dtype)
    q_np, k_np, v_np = _attention_inputs(
        qk_dim=12, v_dim=12, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, 4, 12]
    with pytest.raises(ValueError, match="multiples of 8"):
        eq.stamp(inputs_gpu, stream)
