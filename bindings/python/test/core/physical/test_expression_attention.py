import math

import numpy as np
import pytest
import thor
from thor.physical import (
    AttentionMaskKind,
    AttentionTensorLayout,
    RotaryScalingKind,
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


def _dropout_scalar_gpu(value: int, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    arr = np.asarray([[[[value]]]], dtype=np.int64)
    return _host_to_gpu(arr, thor.DataType.int64, stream, gpu_num=gpu_num)


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
        return (kv_pos > (q_pos - left)) & (kv_pos <= (q_pos + right))
    if kind == AttentionMaskKind.causal_bottom_right:
        offset = kv_len - query_len
        return kv_pos <= (q_pos + offset)
    if kind == AttentionMaskKind.sliding_window_bottom_right:
        offset = kv_len - query_len
        center = q_pos + offset
        return (kv_pos > (center - left)) & (kv_pos <= (center + right))
    raise AssertionError(f"Unhandled mask kind: {kind}")


def _alibi_slopes(num_heads: int) -> np.ndarray:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2.0 ** (-8.0 / float(closest_power_of_2))
    slopes = np.power(base, np.arange(1, closest_power_of_2 + 1, dtype=np.float32)).astype(np.float32)
    if closest_power_of_2 < num_heads:
        extra_base = 2.0 ** (-4.0 / float(closest_power_of_2))
        extra_count = num_heads - closest_power_of_2
        extra_exponents = np.arange(1, 1 + 2 * extra_count, 2, dtype=np.float32)
        slopes = np.concatenate([slopes, np.power(extra_base, extra_exponents).astype(np.float32)])
    return slopes.astype(np.float32)


def _alibi_bias(num_heads: int, query_len: int, kv_len: int) -> np.ndarray:
    q_pos = np.arange(query_len, dtype=np.float32)[:, None]
    kv_pos = np.arange(kv_len, dtype=np.float32)[None, :]
    distance = kv_pos - q_pos
    slopes = _alibi_slopes(num_heads).reshape(1, num_heads, 1, 1)
    return distance.reshape(1, 1, query_len, kv_len).astype(np.float32) * slopes


def _attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | None = None,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    diagonal_left_bound: int = 0,
    diagonal_right_bound: int = 0,
    bias: np.ndarray | None = None,
    use_alibi_mask: bool = False,
    q_seq_len: np.ndarray | None = None,
    kv_seq_len: np.ndarray | None = None,
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
    if bias is not None:
        scores = scores + bias.astype(np.float32)
    if use_alibi_mask:
        scores = scores + _alibi_bias(query_heads, query_len, kv_len)

    mask = _mask_for(mask_kind, query_len, kv_len, diagonal_left_bound, diagonal_right_bound)
    if mask is not None:
        scores = np.where(mask[None, None, :, :], scores, np.float32(-1.0e30))
    if q_seq_len is not None or kv_seq_len is not None:
        assert q_seq_len is not None and kv_seq_len is not None
        q_valid = np.arange(query_len)[None, :] < q_seq_len.astype(np.int64)[:, None]
        kv_valid = np.arange(kv_len)[None, :] < kv_seq_len.astype(np.int64)[:, None]
        padding_mask = q_valid[:, None, :, None] & kv_valid[:, None, None, :]
        scores = np.where(padding_mask, scores, np.float32(-1.0e30))

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores).astype(np.float32)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = np.einsum("bhst,bhtd->bhsd", probs, v_expanded).astype(np.float32)
    if q_seq_len is not None:
        q_valid = np.arange(query_len)[None, :] < q_seq_len.astype(np.int64)[:, None]
        out = np.where(q_valid[:, None, :, None], out, np.float32(0.0))
    return out


def _compile_attention(
    *,
    dtype: thor.DataType,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    attention_scale: float | None = None,
    diagonal_left_bound: int = 0,
    diagonal_right_bound: int = 0,
    use_alibi_mask: bool = False,
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
            use_alibi_mask=use_alibi_mask,
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
def test_attention_forward_alibi_causal_top_left_mask_matches_reference_and_differs_from_plain_causal():
    dtype = thor.DataType.fp16
    scale = 0.67 / math.sqrt(16.0)
    eq = _compile_attention(
        dtype=dtype,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        use_alibi_mask=True,
    )
    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=4, kv_heads=4, query_len=6, kv_len=6, dtype=dtype)
    expected = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
    )
    plain_causal_expected = _attention_reference(q_np, k_np, v_np, scale=scale, mask_kind=AttentionMaskKind.causal_top_left)
    assert not np.allclose(expected.astype(np.float32), plain_causal_expected.astype(np.float32), rtol=1e-3, atol=1e-3)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 4, 6, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_alibi_with_padding_and_additive_bias_matches_reference():
    dtype = thor.DataType.fp16
    scale = 0.71 / math.sqrt(16.0)
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        use_alibi_mask=True,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=4, kv_heads=4, query_len=8, kv_len=8, dtype=dtype)
    rng = np.random.default_rng(889)
    bias_np = rng.normal(0.0, 0.15, size=(2, 4, 8, 8)).astype(np.float32)
    q_len_np = np.asarray([8, 5], dtype=np.int32)
    kv_len_np = np.asarray([8, 6], dtype=np.int32)
    expected = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        bias=bias_np,
        use_alibi_mask=True,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4, 8, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


def test_attention_alibi_requires_causal_diagonal_masking_with_zero_right_bound():
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")

    with pytest.raises(RuntimeError, match="use_alibi_mask requires causal diagonal masking"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            use_alibi_mask=True,
            output_dtype=thor.DataType.fp16,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="diagonal_right_bound == 0"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            mask_kind=AttentionMaskKind.sliding_window_top_left,
            diagonal_left_bound=4,
            diagonal_right_bound=1,
            use_alibi_mask=True,
            output_dtype=thor.DataType.fp16,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_forward_causal_bottom_right_decode_mask_matches_reference_and_differs_from_top_left():
    dtype = thor.DataType.fp16
    scale = 0.73 / math.sqrt(16.0)
    eq = _compile_attention(dtype=dtype, mask_kind=AttentionMaskKind.causal_bottom_right, attention_scale=scale)
    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=3, kv_len=9, dtype=dtype)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, mask_kind=AttentionMaskKind.causal_bottom_right)
    top_left_expected = _attention_reference(q_np, k_np, v_np, scale=scale, mask_kind=AttentionMaskKind.causal_top_left)
    assert not np.allclose(expected.astype(np.float32), top_left_expected.astype(np.float32), rtol=1e-3, atol=1e-3)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 2, 3, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_sliding_window_bottom_right_decode_mask_matches_reference():
    dtype = thor.DataType.fp16
    scale = 0.69 / math.sqrt(16.0)
    eq = _compile_attention(
        dtype=dtype,
        mask_kind=AttentionMaskKind.sliding_window_bottom_right,
        diagonal_left_bound=2,
        diagonal_right_bound=1,
        attention_scale=scale,
    )
    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=10, dtype=dtype)
    expected = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        mask_kind=AttentionMaskKind.sliding_window_bottom_right,
        diagonal_left_bound=2,
        diagonal_right_bound=1,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 2, 4, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_padding_mask_seq_lengths_match_reference_and_stays_single_stage():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    scale = 0.6 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_len_np = np.asarray([6, 4], dtype=np.int32)
    kv_len_np = np.asarray([7, 5], dtype=np.int32)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_len_np, kv_seq_len=kv_len_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, 6, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_padding_mask_bf16_seq_lengths_match_reference():
    dtype = thor.DataType.bf16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    scale = 0.5 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_len_np = np.asarray([5, 3], dtype=np.int32)
    kv_len_np = np.asarray([6, 4], dtype=np.int32)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_len_np, kv_seq_len=kv_len_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, 6, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_padding_mask_requires_int32_seq_lengths():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    eq = ex.compile(
        ex.scaled_dot_product_attention(
            q, k, v, q_seq_len=q_seq_len, kv_seq_len=kv_seq_len, output_dtype=dtype, compute_dtype=thor.DataType.fp32),
        device_num=0,
    )

    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    bad_len = np.asarray([4], dtype=np.float32)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(bad_len, thor.DataType.fp32, stream),
        "kv_seq_len": _host_to_gpu(bad_len, thor.DataType.fp32, stream),
    }

    with pytest.raises(RuntimeError, match="INT32"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_attention_padding_mask_requires_q_and_kv_seq_lengths_together():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")

    with pytest.raises(RuntimeError, match="provided together"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_padding_mask_requires_seq_lengths_shape_batch():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    eq = ex.compile(
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ),
        device_num=0,
    )

    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    bad_len = np.asarray([[4]], dtype=np.int32)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(bad_len, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(bad_len, thor.DataType.int32, stream),
    }

    with pytest.raises(RuntimeError, match=r"shape \[B\]"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_attention_padding_mask_requires_seq_length_extent_to_match_batch():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    eq = ex.compile(
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ),
        device_num=0,
    )

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    bad_len = np.asarray([4], dtype=np.int32)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(bad_len, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(bad_len, thor.DataType.int32, stream),
    }

    with pytest.raises(RuntimeError, match=r"shape \[B\]"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_attention_forward_padding_mask_with_gqa_and_additive_bias_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    scale = 0.55 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=4, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_len_np = np.asarray([6, 4], dtype=np.int32)
    kv_len_np = np.asarray([7, 5], dtype=np.int32)
    rng = np.random.default_rng(4242)
    bias_np = rng.normal(0.0, 0.2, size=(2, 4, 6, 7)).astype(np.float32)
    expected = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4, 6, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_padding_mask_and_causal_mask_match_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        mask_kind=AttentionMaskKind.causal_top_left,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=6, dtype=dtype)
    q_len_np = np.asarray([6, 4], dtype=np.int32)
    kv_len_np = np.asarray([6, 5], dtype=np.int32)
    expected = _attention_reference(
        q_np,
        k_np,
        v_np,
        mask_kind=AttentionMaskKind.causal_top_left,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, 6, 16]
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
def test_attention_padding_mask_stage_feeds_fused_epilogue_and_multi_output_reuses_stage():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    scale = 0.7 / math.sqrt(16.0)
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    shifted = (attn * 1.125 - 0.0625).with_output_dtype(dtype)
    outputs = ex.outputs({
        "attention": attn,
        "shifted": shifted,
    })
    eq = ex.compile(outputs, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_len_np = np.asarray([6, 4], dtype=np.int32)
    kv_len_np = np.asarray([7, 5], dtype=np.int32)
    expected_attention = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_len_np, kv_seq_len=kv_len_np)
    expected_shifted = expected_attention * np.float32(1.125) - np.float32(0.0625)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    assert eq.output_shapes(inputs_gpu) == {
        "attention": [2, 2, 6, 16],
        "shifted": [2, 2, 6, 16],
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


def _attention_backward_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dO: np.ndarray,
    *,
    scale: float | None = None,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    diagonal_left_bound: int = 0,
    diagonal_right_bound: int = 0,
    bias: np.ndarray | None = None,
    use_alibi_mask: bool = False,
    q_seq_len: np.ndarray | None = None,
    kv_seq_len: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    v32 = v.astype(np.float32)
    dO32 = dO.astype(np.float32)

    batch, query_heads, query_len, qk_dim = q32.shape
    _, kv_heads, kv_len, _ = k32.shape
    _, _, _, v_dim = v32.shape
    repeat = query_heads // kv_heads
    kv_head_for_query_head = np.arange(query_heads) // repeat
    k_expanded = k32[:, kv_head_for_query_head, :, :]
    v_expanded = v32[:, kv_head_for_query_head, :, :]

    effective_scale = np.float32(scale if scale is not None else 1.0 / math.sqrt(float(qk_dim)))
    scores = np.einsum("bhsd,bhtd->bhst", q32, k_expanded) * effective_scale
    if bias is not None:
        scores = scores + bias.astype(np.float32)
    if use_alibi_mask:
        scores = scores + _alibi_bias(query_heads, query_len, kv_len)
    mask = _mask_for(mask_kind, query_len, kv_len, diagonal_left_bound, diagonal_right_bound)
    if mask is not None:
        scores = np.where(mask[None, None, :, :], scores, np.float32(-1.0e30))

    q_valid = None
    padding_mask = None
    if q_seq_len is not None or kv_seq_len is not None:
        assert q_seq_len is not None and kv_seq_len is not None
        q_valid = np.arange(query_len)[None, :] < q_seq_len.astype(np.int64)[:, None]
        kv_valid = np.arange(kv_len)[None, :] < kv_seq_len.astype(np.int64)[:, None]
        padding_mask = q_valid[:, None, :, None] & kv_valid[:, None, None, :]
        scores = np.where(padding_mask, scores, np.float32(-1.0e30))

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores).astype(np.float32)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    if q_valid is not None:
        dO32 = np.where(q_valid[:, None, :, None], dO32, np.float32(0.0))

    dP = np.einsum("bhsd,bhtd->bhst", dO32, v_expanded)
    dS = probs * (dP - np.sum(dP * probs, axis=-1, keepdims=True))
    if mask is not None:
        dS = np.where(mask[None, None, :, :], dS, np.float32(0.0))
    if padding_mask is not None:
        dS = np.where(padding_mask, dS, np.float32(0.0))

    dQ = np.einsum("bhst,bhtd->bhsd", dS, k_expanded) * effective_scale
    dK_expanded = np.einsum("bhst,bhsd->bhtd", dS, q32) * effective_scale
    dV_expanded = np.einsum("bhst,bhsd->bhtd", probs, dO32)

    dK = np.zeros((batch, kv_heads, kv_len, qk_dim), dtype=np.float32)
    dV = np.zeros((batch, kv_heads, kv_len, v_dim), dtype=np.float32)
    for h in range(query_heads):
        kv_h = kv_head_for_query_head[h]
        dK[:, kv_h, :, :] += dK_expanded[:, h, :, :]
        dV[:, kv_h, :, :] += dV_expanded[:, h, :, :]

    return dQ, dK, dV


@pytest.mark.cuda
def test_attention_compile_backward_qkv_uses_single_attention_backward_stage_and_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.9 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["q_grad", "k_grad", "v_grad"]

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(9876)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(q_np, k_np, v_np, dO_np, scale=scale)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_alibi_causal_mask_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.76 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        use_alibi_mask=True,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=4, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(9883)
    dO_np = rng.normal(0.0, 0.25, size=(1, 4, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 4, 64, 64],
        "k_grad": [1, 4, 64, 64],
        "v_grad": [1, 4, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_backward_with_alibi_reuses_same_plan_forward_stats_when_forward_output_is_needed():
    dtype = thor.DataType.fp16
    scale = 0.82 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        use_alibi_mask=True,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=4, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    expected_attn = _attention_reference(
        q_np, k_np, v_np, scale=scale, mask_kind=AttentionMaskKind.causal_top_left, use_alibi_mask=True)
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_causal_bottom_right_decode_mask_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.77 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        mask_kind=AttentionMaskKind.causal_bottom_right,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=32, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(9881)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 32, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_bottom_right,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 32, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_backward_with_causal_bottom_right_decode_mask_reuses_same_plan_forward_stats():
    dtype = thor.DataType.fp16
    scale = 0.81 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        mask_kind=AttentionMaskKind.causal_bottom_right,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=32, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    expected_attn = _attention_reference(q_np, k_np, v_np, scale=scale, mask_kind=AttentionMaskKind.causal_bottom_right)
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_bottom_right,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_padding_mask_stays_single_attention_backward_stage_and_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.65 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_len_np = np.asarray([48], dtype=np.int32)
    kv_len_np = np.asarray([55], dtype=np.int32)
    rng = np.random.default_rng(9877)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_padding_mask_and_additive_bias_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.62 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_len_np = np.asarray([50], dtype=np.int32)
    kv_len_np = np.asarray([57], dtype=np.int32)
    rng = np.random.default_rng(9878)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_padding_mask_and_causal_mask_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.68 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_len_np = np.asarray([49], dtype=np.int32)
    kv_len_np = np.asarray([53], dtype=np.int32)
    rng = np.random.default_rng(9879)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_backward_with_padding_mask_reuses_same_plan_forward_stats_when_forward_output_is_needed():
    dtype = thor.DataType.fp16
    scale = 0.72 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_len_np = np.asarray([52], dtype=np.int32)
    kv_len_np = np.asarray([47], dtype=np.int32)
    expected_attn = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_len_np, kv_seq_len=kv_len_np)
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_backward_with_padding_mask_and_additive_bias_reuses_same_plan_forward_stats():
    dtype = thor.DataType.fp16
    scale = 0.74 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_len_np = np.asarray([51], dtype=np.int32)
    kv_len_np = np.asarray([56], dtype=np.int32)
    rng = np.random.default_rng(9880)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    expected_attn = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_len_np,
        kv_seq_len=kv_len_np,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_backward_reuses_same_plan_forward_stats_when_forward_output_is_needed():
    dtype = thor.DataType.fp16
    scale = 0.8 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    expected_attn = _attention_reference(q_np, k_np, v_np, scale=scale)
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(q_np, k_np, v_np, np.float32(2.0) * expected_attn, scale=scale)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


def _rope_reference(
    x: np.ndarray,
    *,
    sequence_axis: int = 2,
    head_dim_axis: int = 3,
    rotary_dim: int = 0,
    base: float = 10000.0,
    position_offset: int = 0,
    interleaved: bool = False,
    inverse: bool = False,
    scaling_kind: RotaryScalingKind = RotaryScalingKind.none,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 0,
) -> np.ndarray:
    x32 = x.astype(np.float32)
    if x32.ndim != 4 or sequence_axis != 2 or head_dim_axis != 3:
        raise AssertionError("test reference currently covers rank-4 [B,H,S,D] RoPE")

    out = x32.copy()
    seq_len = x32.shape[sequence_axis]
    head_dim = x32.shape[head_dim_axis]
    effective_rotary_dim = head_dim if rotary_dim == 0 else rotary_dim
    if effective_rotary_dim <= 0 or effective_rotary_dim > head_dim or effective_rotary_dim % 2 != 0:
        raise ValueError("invalid rotary_dim")

    rope_base = np.float32(base)
    if scaling_kind == RotaryScalingKind.dynamic_ntk:
        if original_max_position_embeddings <= 0:
            raise ValueError("dynamic_ntk requires original_max_position_embeddings")
        seq_for_ntk = max(float(seq_len + max(0, position_offset)), 1.0)
        if seq_for_ntk > float(original_max_position_embeddings) and effective_rotary_dim > 2:
            ratio = (scaling_factor * seq_for_ntk / float(original_max_position_embeddings)) - (scaling_factor - 1.0)
            rope_base = np.float32(base * (ratio ** (float(effective_rotary_dim) / float(effective_rotary_dim - 2))))

    half = effective_rotary_dim // 2
    pair_indices = np.arange(half, dtype=np.float32)
    inv_freq = rope_base ** (-2.0 * pair_indices / np.float32(effective_rotary_dim))
    positions = np.arange(seq_len, dtype=np.float32) + np.float32(position_offset)
    if scaling_kind == RotaryScalingKind.linear:
        positions = positions / np.float32(scaling_factor)
    theta = positions[:, None] * inv_freq[None, :]
    sin = np.sin(theta).astype(np.float32)
    cos = np.cos(theta).astype(np.float32)
    if inverse:
        sin = -sin

    if interleaved:
        first_idx = np.arange(0, effective_rotary_dim, 2)
        second_idx = np.arange(1, effective_rotary_dim, 2)
    else:
        first_idx = np.arange(half)
        second_idx = np.arange(half, effective_rotary_dim)

    x_first = x32[:, :, :, first_idx]
    x_second = x32[:, :, :, second_idx]
    out[:, :, :, first_idx] = x_first * cos[None, None, :, :] - x_second * sin[None, None, :, :]
    out[:, :, :, second_idx] = x_first * sin[None, None, :, :] + x_second * cos[None, None, :, :]
    return out


@pytest.mark.cuda
def test_rope_forward_standard_and_linear_scaling_match_reference():
    dtype = thor.DataType.fp16
    q_np, _, _ = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=6, kv_len=6, dtype=dtype)
    q = ex.input("q")
    out = ex.rope(
        q,
        rotary_dim=8,
        base=10000.0,
        position_offset=3,
        scaling_kind=RotaryScalingKind.linear,
        scaling_factor=2.0,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)
    expected = _rope_reference(
        q_np,
        rotary_dim=8,
        position_offset=3,
        scaling_kind=RotaryScalingKind.linear,
        scaling_factor=2.0,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {"q": _host_to_gpu(q_np, dtype, stream)}
    assert eq.output_shape(inputs_gpu) == [1, 2, 6, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_rope_forward_interleaved_dynamic_ntk_scaling_matches_reference():
    dtype = thor.DataType.fp16
    q_np, _, _ = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=7, kv_len=7, dtype=dtype)
    q = ex.input("q")
    out = ex.rope(
        q,
        rotary_dim=8,
        interleaved=True,
        scaling_kind=RotaryScalingKind.dynamic_ntk,
        scaling_factor=2.0,
        original_max_position_embeddings=4,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)
    expected = _rope_reference(
        q_np,
        rotary_dim=8,
        interleaved=True,
        scaling_kind=RotaryScalingKind.dynamic_ntk,
        scaling_factor=2.0,
        original_max_position_embeddings=4,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {"q": _host_to_gpu(q_np, dtype, stream)}
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_rope_qk_feed_single_attention_stage_and_match_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_rot = ex.rope(q, rotary_dim=16, output_dtype=dtype, compute_dtype=thor.DataType.fp32)
    k_rot = ex.rope(k, rotary_dim=16, output_dtype=dtype, compute_dtype=thor.DataType.fp32)
    out = ex.scaled_dot_product_attention(q_rot, k_rot, v, output_dtype=dtype, compute_dtype=thor.DataType.fp32)
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=4, dtype=dtype)
    q_expected = _rope_reference(q_np, rotary_dim=16)
    k_expected = _rope_reference(k_np, rotary_dim=16)
    expected = _attention_reference(q_expected, k_expected, v_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
    }
    kinds = eq._debug_stage_kinds(inputs_gpu)
    assert kinds[-1] == "Attention"
    assert kinds.count("Attention") == 1
    assert all(kind == "FusedKernel" for kind in kinds[:-1])

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_rope_compile_backward_is_inverse_rope_and_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    q = ex.input("q")
    out = ex.rope(
        q,
        rotary_dim=8,
        position_offset=2,
        scaling_kind=RotaryScalingKind.linear,
        scaling_factor=1.5,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q"], error_input_name=upstream_name)

    q_np, _, _ = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=5, kv_len=5, dtype=dtype)
    rng = np.random.default_rng(2468)
    dO_np = rng.normal(0.0, 0.2, size=q_np.shape).astype(_numpy_storage_dtype(dtype))
    expected = _rope_reference(
        dO_np,
        rotary_dim=8,
        position_offset=2,
        inverse=True,
        scaling_kind=RotaryScalingKind.linear,
        scaling_factor=1.5,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }
    assert bwd_eq.output_shapes(inputs_gpu) == {"q_grad": [1, 2, 5, 16]}
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.outputs()["q_grad"], dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_rope_invalid_rotary_dim_raises_during_expression_construction():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    with pytest.raises(RuntimeError, match="rotary_dim"):
        ex.rope(q, rotary_dim=7, output_dtype=dtype, compute_dtype=thor.DataType.fp32)


@pytest.mark.cuda
def test_attention_forward_with_additive_bias_matches_reference_and_stays_single_attention_stage():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    scale = 0.8 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=5, dtype=dtype)
    rng = np.random.default_rng(777)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 4, 5)).astype(np.float32)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, bias=bias_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 2, 4, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_additive_bias_stays_single_attention_backward_stage_and_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.7 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(888)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(q_np, k_np, v_np, dO_np, scale=scale, bias=bias_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(_copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(_copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(_copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_forward_dropout_philox_is_deterministic_and_single_attention_stage():
    dtype = thor.DataType.fp16
    scale = 0.77 / math.sqrt(64.0)
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        attention_scale=scale,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "dropout_seed": _dropout_scalar_gpu(1234, stream),
        "dropout_offset": _dropout_scalar_gpu(5678, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 2, 64, 64]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped1 = eq.stamp(inputs_gpu, stream)
    stamped1.run()
    got1 = _copy_to_host(stamped1.output(), dtype, stream)

    stamped2 = eq.stamp(inputs_gpu, stream)
    stamped2.run()
    got2 = _copy_to_host(stamped2.output(), dtype, stream)
    np.testing.assert_array_equal(got1, got2)

    inputs_gpu_different_offset = dict(inputs_gpu)
    inputs_gpu_different_offset["dropout_offset"] = _dropout_scalar_gpu(5679, stream)
    stamped3 = eq.stamp(inputs_gpu_different_offset, stream)
    stamped3.run()
    got3 = _copy_to_host(stamped3.output(), dtype, stream)
    assert np.max(np.abs(got1.astype(np.float32) - got3.astype(np.float32))) > 1.0e-3


@pytest.mark.cuda
def test_attention_dropout_requires_seed_offset_together_and_probability_enabled():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")

    with pytest.raises(RuntimeError, match="dropout_probability"):
        ex.scaled_dot_product_attention(
            q, k, v, dropout_probability=0.25, output_dtype=dtype, compute_dtype=thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="dropout_seed and dropout_offset"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_probability=0.25,
            dropout_seed=dropout_seed,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="dropout_probability is zero"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_seed=dropout_seed,
            dropout_offset=dropout_offset,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_dropout_requires_int64_scalar_seed_offset():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    eq = ex.compile(
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_probability=0.5,
            dropout_seed=dropout_seed,
            dropout_offset=dropout_offset,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ),
        device_num=0,
    )

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_bad_dtype = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "dropout_seed": _host_to_gpu(np.asarray([[[[1234]]]], dtype=np.int32), thor.DataType.int32, stream),
        "dropout_offset": _dropout_scalar_gpu(5678, stream),
    }
    with pytest.raises(RuntimeError, match="dropout seed dtype must be INT64"):
        eq.stamp(inputs_bad_dtype, stream)

    inputs_bad_shape = dict(inputs_bad_dtype)
    inputs_bad_shape["dropout_seed"] = _host_to_gpu(np.asarray([1234], dtype=np.int64), thor.DataType.int64, stream)
    inputs_bad_shape["dropout_offset"] = _dropout_scalar_gpu(5678, stream)
    with pytest.raises(RuntimeError, match=r"dropout seed shape must be \[1,1,1,1\]"):
        eq.stamp(inputs_bad_shape, stream)


@pytest.mark.cuda
def test_attention_forward_dropout_with_padding_and_bias_stays_single_stage_and_is_deterministic():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(4321)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    q_len_np = np.asarray([53], dtype=np.int32)
    kv_len_np = np.asarray([49], dtype=np.int32)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
        "dropout_seed": _dropout_scalar_gpu(1001, stream),
        "dropout_offset": _dropout_scalar_gpu(2002, stream),
    }

    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped1 = eq.stamp(inputs_gpu, stream)
    stamped1.run()
    got1 = _copy_to_host(stamped1.output(), dtype, stream)
    stamped2 = eq.stamp(inputs_gpu, stream)
    stamped2.run()
    got2 = _copy_to_host(stamped2.output(), dtype, stream)
    np.testing.assert_array_equal(got1, got2)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_dropout_stays_single_attention_backward_stage():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(2468)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "dropout_seed": _dropout_scalar_gpu(4242, stream),
        "dropout_offset": _dropout_scalar_gpu(3434, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    assert got["q_grad"].dimensions == [1, 2, 64, 64]
    assert got["k_grad"].dimensions == [1, 2, 64, 64]
    assert got["v_grad"].dimensions == [1, 2, 64, 64]


@pytest.mark.cuda
def test_attention_backward_with_dropout_reuses_same_plan_forward_stats_when_forward_output_is_needed():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "dropout_seed": _dropout_scalar_gpu(5151, stream),
        "dropout_offset": _dropout_scalar_gpu(6161, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    assert got["q_grad"].dimensions == [1, 2, 64, 64]
    assert got["k_grad"].dimensions == [1, 2, 64, 64]
    assert got["v_grad"].dimensions == [1, 2, 64, 64]


@pytest.mark.cuda
def test_attention_forward_dropout_with_alibi_causal_mask_stays_single_stage_and_is_deterministic():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=4, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "dropout_seed": _dropout_scalar_gpu(7070, stream),
        "dropout_offset": _dropout_scalar_gpu(8080, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 4, 64, 64]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped1 = eq.stamp(inputs_gpu, stream)
    stamped1.run()
    got1 = _copy_to_host(stamped1.output(), dtype, stream)
    stamped2 = eq.stamp(inputs_gpu, stream)
    stamped2.run()
    got2 = _copy_to_host(stamped2.output(), dtype, stream)
    np.testing.assert_array_equal(got1, got2)


@pytest.mark.cuda
def test_attention_rejects_bottom_right_decode_mask_with_dropout_before_cudnn_graph_build():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")

    with pytest.raises(RuntimeError, match="CausalBottomRight.*dropout"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            mask_kind=AttentionMaskKind.causal_bottom_right,
            dropout_probability=0.5,
            dropout_seed=dropout_seed,
            dropout_offset=dropout_offset,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_rejects_bottom_right_decode_mask_with_additive_bias_before_cudnn_graph_build():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")

    with pytest.raises(RuntimeError, match="CausalBottomRight.*additive bias"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            bias=bias,
            mask_kind=AttentionMaskKind.causal_bottom_right,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_dropout_padding_and_bias_stays_single_attention_backward_stage():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(9741)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    q_len_np = np.asarray([57], dtype=np.int32)
    kv_len_np = np.asarray([51], dtype=np.int32)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_len_np, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_len_np, thor.DataType.int32, stream),
        "dropout_seed": _dropout_scalar_gpu(1111, stream),
        "dropout_offset": _dropout_scalar_gpu(2222, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    assert got["q_grad"].dimensions == [1, 2, 64, 64]
    assert got["k_grad"].dimensions == [1, 2, 64, 64]
    assert got["v_grad"].dimensions == [1, 2, 64, 64]


@pytest.mark.cuda
def test_attention_backward_with_dropout_alibi_reuses_same_plan_forward_stats_when_forward_output_is_needed():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=4, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "dropout_seed": _dropout_scalar_gpu(1212, stream),
        "dropout_offset": _dropout_scalar_gpu(3434, stream),
    }

    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    assert got["q_grad"].dimensions == [1, 4, 64, 64]
    assert got["k_grad"].dimensions == [1, 4, 64, 64]
    assert got["v_grad"].dimensions == [1, 4, 64, 64]
