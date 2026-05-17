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


def _ragged_element_offsets(lengths: np.ndarray, heads: int, dim: int) -> np.ndarray:
    element_offsets = np.zeros((len(lengths) + 1,), dtype=np.int32)
    element_offsets[1:] = np.cumsum(lengths.astype(np.int64) * np.int64(heads) * np.int64(dim)).astype(np.int32)
    return element_offsets




def _pack_bshd_dense_storage(logical_bhsd: np.ndarray) -> np.ndarray:
    # Thor-side BSHD tensors are actually shaped [B,S,H,D].  References stay in
    # semantic BHSD order, so convert reference values to the Thor tensor shape.
    return np.ascontiguousarray(logical_bhsd.transpose(0, 2, 1, 3))


def _pack_bshd_ragged_storage(logical_bhsd: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    # Ragged cuDNN offsets index token-contiguous THD storage.  The Thor tensor
    # shape is still [B,S,H,D], but valid tokens from all batch items are packed
    # at the front of the flat buffer according to the supplied offsets.
    batch, heads, sequence_length, head_dim = logical_bhsd.shape
    packed = np.zeros((batch, sequence_length, heads, head_dim), dtype=logical_bhsd.dtype)
    flat = packed.reshape(-1)
    cursor = 0
    for b in range(batch):
        valid = int(lengths[b])
        token_contiguous = logical_bhsd[b, :, :valid, :].transpose(1, 0, 2).reshape(-1)
        flat[cursor:cursor + token_contiguous.size] = token_contiguous
        cursor += token_contiguous.size
    return packed


def _packed_bshd_ragged_valid_values(storage: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    batch, _, heads, head_dim = storage.shape
    offsets = _ragged_element_offsets(lengths, heads, head_dim)
    flat = storage.reshape(-1)
    pieces = [flat[int(offsets[b]):int(offsets[b + 1])] for b in range(batch)]
    return np.concatenate(pieces) if pieces else np.asarray([], dtype=storage.dtype)


def _assert_packed_bshd_ragged_close(got: np.ndarray, expected: np.ndarray, lengths: np.ndarray):
    np.testing.assert_allclose(
        _packed_bshd_ragged_valid_values(got, lengths).astype(np.float32),
        _packed_bshd_ragged_valid_values(expected, lengths).astype(np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


def _copy_to_host(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


def _dropout_scalar_gpu(value: int, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    arr = np.asarray([[[[value]]]], dtype=np.int64)
    return _host_to_gpu(arr, thor.DataType.int64, stream, gpu_num=gpu_num)


def _page_table(batch: int, kv_len: int, block_size: int) -> np.ndarray:
    pages = int(math.ceil(float(kv_len) / float(block_size)))
    return np.arange(batch * pages, dtype=np.int32).reshape(batch, 1, pages, 1)


def _paged_kv_container_from_page_table(logical: np.ndarray, block_size: int, page_table: np.ndarray) -> np.ndarray:
    batch, heads, sequence_length, head_dim = logical.shape
    pages = int(math.ceil(float(sequence_length) / float(block_size)))
    assert page_table.shape == (batch, 1, pages, 1)
    max_block_id = int(np.max(page_table))
    container = np.zeros((max_block_id + 1, heads, block_size, head_dim), dtype=logical.dtype)
    for b in range(batch):
        for page in range(pages):
            block_id = int(page_table[b, 0, page, 0])
            src_begin = page * block_size
            src_end = min(src_begin + block_size, sequence_length)
            valid = src_end - src_begin
            if valid > 0:
                container[block_id, :, :valid, :] = logical[b, :, src_begin:src_end, :]
    return container


def _paged_kv_container(logical: np.ndarray, block_size: int) -> np.ndarray:
    batch, _, sequence_length, _ = logical.shape
    return _paged_kv_container_from_page_table(logical, block_size, _page_table(batch, sequence_length, block_size))


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


def _bottom_right_mask_for_batch(
    kind: AttentionMaskKind,
    query_len: int,
    kv_len: int,
    q_seq_len: np.ndarray,
    kv_seq_len: np.ndarray,
    left: int = 0,
    right: int = 0,
) -> np.ndarray:
    q_pos = np.arange(query_len)[None, :, None]
    kv_pos = np.arange(kv_len)[None, None, :]
    q_valid_len = q_seq_len.astype(np.int64)[:, None, None]
    kv_valid_len = kv_seq_len.astype(np.int64)[:, None, None]
    center = q_pos + (kv_valid_len - q_valid_len)

    if kind == AttentionMaskKind.causal_bottom_right:
        return kv_pos <= center
    if kind == AttentionMaskKind.sliding_window_bottom_right:
        # cuDNN's sliding-window left bound is exclusive.  For bottom-right decode masks,
        # the diagonal is anchored to each batch item's effective q/kv sequence lengths,
        # not to the padded maximum tensor extents.
        return (kv_pos > (center - left)) & (kv_pos <= (center + right))
    raise AssertionError(f"Unhandled bottom-right mask kind: {kind}")


def _alibi_slopes(num_heads: int) -> np.ndarray:
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    base = 2.0**(-8.0 / float(closest_power_of_2))
    slopes = np.power(base, np.arange(1, closest_power_of_2 + 1, dtype=np.float32)).astype(np.float32)
    if closest_power_of_2 < num_heads:
        extra_base = 2.0**(-4.0 / float(closest_power_of_2))
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

    if (mask_kind in (AttentionMaskKind.causal_bottom_right, AttentionMaskKind.sliding_window_bottom_right) and
            q_seq_len is not None and kv_seq_len is not None):
        mask = _bottom_right_mask_for_batch(
            mask_kind, query_len, kv_len, q_seq_len, kv_seq_len, diagonal_left_bound, diagonal_right_bound)
        scores = np.where(mask[:, None, :, :], scores, np.float32(-1.0e30))
    else:
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
def test_expression_reshape_of_matmul_stage_output_is_metadata_alias_not_fused_kernel():
    dtype = thor.DataType.fp16
    a = ex.input("a")
    b = ex.input("b")
    out = ex.matmul(a, b, compute_dtype=thor.DataType.fp32, output_dtype=dtype).reshape([2, 3, 5])
    eq = ex.compile(out, device_num=0)

    rng = np.random.default_rng(7011)
    a_np = rng.normal(0.0, 0.25, size=(6, 4)).astype(_numpy_storage_dtype(dtype))
    b_np = rng.normal(0.0, 0.25, size=(4, 5)).astype(_numpy_storage_dtype(dtype))
    expected = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(_numpy_storage_dtype(dtype)).reshape(2, 3, 5)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    kinds = eq._debug_stage_kinds(inputs_gpu)
    assert len(kinds) == 1
    assert kinds[0].startswith("Matmul")
    assert "FusedKernel" not in kinds
    assert eq.output_shape(inputs_gpu) == [2, 3, 5]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_expression_reshape_with_same_output_dtype_is_metadata_alias_not_fused_kernel():
    dtype = thor.DataType.fp16
    a = ex.input("a")
    b = ex.input("b")
    out = ex.matmul(a, b, compute_dtype=thor.DataType.fp32, output_dtype=dtype).reshape([2, 3, 5]).with_output_dtype(dtype)
    eq = ex.compile(out, device_num=0)

    rng = np.random.default_rng(7012)
    a_np = rng.normal(0.0, 0.25, size=(6, 4)).astype(_numpy_storage_dtype(dtype))
    b_np = rng.normal(0.0, 0.25, size=(4, 5)).astype(_numpy_storage_dtype(dtype))
    expected = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(_numpy_storage_dtype(dtype)).reshape(2, 3, 5)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    kinds = eq._debug_stage_kinds(inputs_gpu)
    assert len(kinds) == 1
    assert kinds[0].startswith("Matmul")
    assert "FusedKernel" not in kinds
    assert eq.output_shape(inputs_gpu) == [2, 3, 5]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)



@pytest.mark.cuda
def test_composed_attention_projection_biases_lower_to_matmul_bias_epilogues_without_fused_adds():
    dtype = thor.DataType.fp16
    batch = 2
    sequence = 4
    input_features = 32
    heads = 2
    kv_heads = 2
    head_dim = 16
    value_dim = 16
    output_features = 32
    scale = 0.5 / math.sqrt(float(head_dim))

    x = ex.input("x")
    qw = ex.input("qw")
    kw = ex.input("kw")
    vw = ex.input("vw")
    ow = ex.input("ow")
    qb = ex.input("qb")
    kb = ex.input("kb")
    vb = ex.input("vb")
    ob = ex.input("ob")

    flat = x.reshape([batch * sequence, input_features])
    q = (ex.matmul(flat, qw, compute_dtype=thor.DataType.fp32, output_dtype=dtype) + qb).reshape(
        [batch, sequence, heads, head_dim])
    k = (ex.matmul(flat, kw, compute_dtype=thor.DataType.fp32, output_dtype=dtype) + kb).reshape(
        [batch, sequence, kv_heads, head_dim])
    v = (ex.matmul(flat, vw, compute_dtype=thor.DataType.fp32, output_dtype=dtype) + vb).reshape(
        [batch, sequence, kv_heads, value_dim])
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    merged = attn.reshape([batch * sequence, heads * value_dim])
    out = (ex.matmul(merged, ow, compute_dtype=thor.DataType.fp32, output_dtype=dtype) + ob).reshape(
        [batch, sequence, output_features])
    eq = ex.compile(out, device_num=0)

    rng = np.random.default_rng(8917)
    storage_dtype = _numpy_storage_dtype(dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(rng.normal(0.0, 0.3, size=(batch, sequence, input_features)).astype(storage_dtype), dtype, stream),
        "qw": _host_to_gpu(rng.normal(0.0, 0.2, size=(input_features, heads * head_dim)).astype(storage_dtype), dtype, stream),
        "kw": _host_to_gpu(rng.normal(0.0, 0.2, size=(input_features, kv_heads * head_dim)).astype(storage_dtype), dtype, stream),
        "vw": _host_to_gpu(rng.normal(0.0, 0.2, size=(input_features, kv_heads * value_dim)).astype(storage_dtype), dtype, stream),
        "ow": _host_to_gpu(rng.normal(0.0, 0.2, size=(heads * value_dim, output_features)).astype(storage_dtype), dtype, stream),
        "qb": _host_to_gpu(rng.normal(0.0, 0.05, size=(heads * head_dim,)).astype(storage_dtype), dtype, stream),
        "kb": _host_to_gpu(rng.normal(0.0, 0.05, size=(kv_heads * head_dim,)).astype(storage_dtype), dtype, stream),
        "vb": _host_to_gpu(rng.normal(0.0, 0.05, size=(kv_heads * value_dim,)).astype(storage_dtype), dtype, stream),
        "ob": _host_to_gpu(rng.normal(0.0, 0.05, size=(output_features,)).astype(storage_dtype), dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [batch, sequence, output_features]
    kinds = eq._debug_stage_kinds(inputs_gpu)
    assert sum(1 for kind in kinds if kind.startswith("Matmul")) == 4
    assert kinds.count("Attention") == 1
    assert "FusedKernel" not in kinds


@pytest.mark.cuda
def test_expression_reshape_inputs_to_attention_stage_are_metadata_aliases_not_fused_kernels():
    dtype = thor.DataType.fp16
    batch = 2
    heads = 2
    sequence = 4
    dim = 16
    scale = 0.77 / math.sqrt(float(dim))

    # BSHD now means the Thor tensor is actually shaped [B,S,H,D].  Reshaping
    # the flat projection output into that layout is a metadata alias and must
    # not introduce a fused/materialization kernel.
    q = ex.input("q_flat").reshape([batch, sequence, heads, dim])
    k = ex.input("k_flat").reshape([batch, sequence, heads, dim])
    v = ex.input("v_flat").reshape([batch, sequence, heads, dim])
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_bhsd, k_bhsd, v_bhsd = _attention_inputs(
        batch=batch, query_heads=heads, kv_heads=heads, query_len=sequence, kv_len=sequence, qk_dim=dim, v_dim=dim, dtype=dtype)
    q_storage = _pack_bshd_dense_storage(q_bhsd)
    k_storage = _pack_bshd_dense_storage(k_bhsd)
    v_storage = _pack_bshd_dense_storage(v_bhsd)
    expected_storage = _pack_bshd_dense_storage(_cast_reference_to_storage_dtype(_attention_reference(q_bhsd, k_bhsd, v_bhsd, scale=scale), dtype))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q_flat": _host_to_gpu(q_storage.reshape(batch * sequence, heads * dim), dtype, stream),
        "k_flat": _host_to_gpu(k_storage.reshape(batch * sequence, heads * dim), dtype, stream),
        "v_flat": _host_to_gpu(v_storage.reshape(batch * sequence, heads * dim), dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [batch, sequence, heads, dim]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected_storage, dtype)




@pytest.mark.cuda
@pytest.mark.parametrize("layout", [AttentionTensorLayout.bhsd, AttentionTensorLayout.bshd])
@pytest.mark.parametrize(
    "query_heads,kv_heads",
    [
        (4, 4),  # MHA
        (4, 2),  # GQA
        (4, 1),  # MQA
    ],
)
def test_attention_dense_layout_contract_matches_reference_for_mha_gqa_mqa(
    layout: AttentionTensorLayout, query_heads: int, kv_heads: int
):
    dtype = thor.DataType.fp16
    batch = 2
    query_len = 3
    kv_len = 5
    qk_dim = 16
    v_dim = 16
    scale = 0.61 / math.sqrt(float(qk_dim))
    eq = _compile_attention(
        dtype=dtype,
        attention_scale=scale,
        q_layout=layout,
        k_layout=layout,
        v_layout=layout,
        o_layout=layout,
    )

    q_np, k_np, v_np = _attention_inputs(
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_len=query_len,
        kv_len=kv_len,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=dtype,
    )
    expected_logical = _cast_reference_to_storage_dtype(_attention_reference(q_np, k_np, v_np, scale=scale), dtype)

    if layout == AttentionTensorLayout.bshd:
        q_storage = _pack_bshd_dense_storage(q_np)
        k_storage = _pack_bshd_dense_storage(k_np)
        v_storage = _pack_bshd_dense_storage(v_np)
        expected_storage = _pack_bshd_dense_storage(expected_logical)
        wrong_layout_storage = expected_logical
    else:
        q_storage = q_np
        k_storage = k_np
        v_storage = v_np
        expected_storage = expected_logical
        wrong_layout_storage = _pack_bshd_dense_storage(expected_logical)

    assert not np.allclose(
        expected_storage.reshape(-1).astype(np.float32),
        wrong_layout_storage.reshape(-1).astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    ), "layout sentinel is degenerate; this test would not catch BHSD/BSHD mixups"

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
    }

    expected_shape = [batch, query_heads, query_len, v_dim]
    if layout == AttentionTensorLayout.bshd:
        expected_shape = [batch, query_len, query_heads, v_dim]
    assert eq.output_shape(inputs_gpu) == expected_shape
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected_storage, dtype)


@pytest.mark.cuda
def test_attention_bshd_strided_packed_qkv_view_backward_scatter_add_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    batch = 2
    sequence = 3
    query_heads = 4
    kv_heads = 2
    qk_dim = 16
    v_dim = 16
    scale = 0.61 / math.sqrt(float(qk_dim))

    q_width = query_heads * qk_dim
    k_width = kv_heads * qk_dim
    v_width = kv_heads * v_dim
    total_width = q_width + k_width + v_width

    qkv = ex.input("qkv")
    q = qkv.strided_view(
        [batch, sequence, query_heads, qk_dim],
        [sequence * total_width, total_width, qk_dim, 1],
        0,
    )
    k = qkv.strided_view(
        [batch, sequence, kv_heads, qk_dim],
        [sequence * total_width, total_width, qk_dim, 1],
        q_width,
    )
    v = qkv.strided_view(
        [batch, sequence, kv_heads, v_dim],
        [sequence * total_width, total_width, v_dim, 1],
        q_width + k_width,
    )
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(attn, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["qkv"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["qkv_grad"]

    q_np, k_np, v_np = _attention_inputs(
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_len=sequence,
        kv_len=sequence,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=dtype,
    )
    q_storage = _pack_bshd_dense_storage(q_np)
    k_storage = _pack_bshd_dense_storage(k_np)
    v_storage = _pack_bshd_dense_storage(v_np)

    qkv_np = np.zeros((batch * sequence, total_width), dtype=_numpy_storage_dtype(dtype))
    qkv_np[:, 0:q_width] = q_storage.reshape(batch * sequence, q_width)
    qkv_np[:, q_width:q_width + k_width] = k_storage.reshape(batch * sequence, k_width)
    qkv_np[:, q_width + k_width:] = v_storage.reshape(batch * sequence, v_width)

    rng = np.random.default_rng(271828)
    dO_np = rng.normal(0.0, 0.25, size=(batch, query_heads, sequence, v_dim)).astype(_numpy_storage_dtype(dtype))
    dO_storage = _pack_bshd_dense_storage(dO_np)
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(q_np, k_np, v_np, dO_np, scale=scale)

    expected_qkv_grad = np.zeros((batch * sequence, total_width), dtype=np.float32)
    expected_qkv_grad[:, 0:q_width] = _pack_bshd_dense_storage(expected_dq).reshape(batch * sequence, q_width)
    expected_qkv_grad[:, q_width:q_width + k_width] = _pack_bshd_dense_storage(expected_dk).reshape(batch * sequence, k_width)
    expected_qkv_grad[:, q_width + k_width:] = _pack_bshd_dense_storage(expected_dv).reshape(batch * sequence, v_width)
    expected_qkv_grad = _cast_reference_to_storage_dtype(expected_qkv_grad, dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "qkv": _host_to_gpu(qkv_np, dtype, stream),
        upstream_name: _host_to_gpu(dO_storage, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {"qkv_grad": [batch * sequence, total_width]}
    stage_kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert stage_kinds.count("AttentionBackward") == 1
    assert stage_kinds.count("FusedKernel") == 1
    assert stage_kinds.index("AttentionBackward") < stage_kinds.index("FusedKernel")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    runtime_stage_kinds = stamped._debug_stage_kinds()
    assert runtime_stage_kinds == ["AttentionBackward"]
    assert "FusedKernel" not in runtime_stage_kinds

    stamped.run()
    got = _copy_to_host(stamped.outputs()["qkv_grad"], dtype, stream)
    _assert_close(got, expected_qkv_grad, dtype)


@pytest.mark.cuda
def test_attention_bshd_strided_packed_qkv_views_match_reference_without_split_kernel():
    dtype = thor.DataType.fp16
    batch = 2
    sequence = 3
    query_heads = 4
    kv_heads = 2
    qk_dim = 16
    v_dim = 16
    scale = 0.73 / math.sqrt(float(qk_dim))

    q_width = query_heads * qk_dim
    k_width = kv_heads * qk_dim
    v_width = kv_heads * v_dim
    total_width = q_width + k_width + v_width

    qkv = ex.input("qkv")
    q = qkv.strided_view(
        [batch, sequence, query_heads, qk_dim],
        [sequence * total_width, total_width, qk_dim, 1],
        0,
    )
    k = qkv.strided_view(
        [batch, sequence, kv_heads, qk_dim],
        [sequence * total_width, total_width, qk_dim, 1],
        q_width,
    )
    v = qkv.strided_view(
        [batch, sequence, kv_heads, v_dim],
        [sequence * total_width, total_width, v_dim, 1],
        q_width + k_width,
    )
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(attn, device_num=0)

    q_np, k_np, v_np = _attention_inputs(
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_len=sequence,
        kv_len=sequence,
        qk_dim=qk_dim,
        v_dim=v_dim,
        dtype=dtype,
    )
    q_storage = _pack_bshd_dense_storage(q_np).reshape(batch, sequence, q_width)
    k_storage = _pack_bshd_dense_storage(k_np).reshape(batch, sequence, k_width)
    v_storage = _pack_bshd_dense_storage(v_np).reshape(batch, sequence, v_width)

    qkv_np = np.zeros((batch, sequence, total_width), dtype=_numpy_storage_dtype(dtype))
    qkv_np[:, :, 0:q_width] = q_storage
    qkv_np[:, :, q_width:q_width + k_width] = k_storage
    qkv_np[:, :, q_width + k_width:] = v_storage
    qkv_np = np.ascontiguousarray(qkv_np.reshape(batch * sequence, total_width))

    expected = _pack_bshd_dense_storage(
        _cast_reference_to_storage_dtype(_attention_reference(q_np, k_np, v_np, scale=scale), dtype)
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {"qkv": _host_to_gpu(qkv_np, dtype, stream)}

    assert eq.output_shape(inputs_gpu) == [batch, sequence, query_heads, v_dim]
    # Q/K/V are storage aliases into one packed input, so no split/materialize stage should be planned.
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_composed_projection_to_attention_bshd_layout_contract_matches_reference():
    dtype = thor.DataType.fp16
    batch = 2
    sequence = 3
    heads = 2
    kv_heads = 2
    head_dim = 16
    value_dim = 16
    input_features = heads * value_dim
    scale = 1.0

    x = ex.input("x")
    qw = ex.input("qw")
    kw = ex.input("kw")
    vw = ex.input("vw")

    flat = x.reshape([batch * sequence, input_features])
    q = ex.matmul(flat, qw, compute_dtype=thor.DataType.fp32, output_dtype=dtype).reshape(
        [batch, sequence, heads, head_dim]
    )
    k = ex.matmul(flat, kw, compute_dtype=thor.DataType.fp32, output_dtype=dtype).reshape(
        [batch, sequence, kv_heads, head_dim]
    )
    v = ex.matmul(flat, vw, compute_dtype=thor.DataType.fp32, output_dtype=dtype).reshape(
        [batch, sequence, kv_heads, value_dim]
    )
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    merged = attn.reshape([batch * sequence, heads * value_dim])
    eq = ex.compile(merged, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.zeros((batch, sequence, input_features), dtype=np.float32)
    for b in range(batch):
        for s in range(sequence):
            for h in range(heads):
                for d in range(value_dim):
                    # Deliberately non-symmetric across batch, token, head, and dim.
                    x_np[b, s, h * value_dim + d] = 0.25 * (b + 1) + 0.10 * s + 0.03 * h + 0.001 * d
    x_np = x_np.astype(storage_dtype)

    qw_np = np.zeros((input_features, heads * head_dim), dtype=storage_dtype)
    kw_np = np.zeros((input_features, kv_heads * head_dim), dtype=storage_dtype)
    vw_np = np.zeros((input_features, kv_heads * value_dim), dtype=np.float32)
    for i in range(kv_heads * value_dim):
        vw_np[i, i] = 1.0
    vw_np = vw_np.astype(storage_dtype)

    q_ref = np.zeros((batch, heads, sequence, head_dim), dtype=storage_dtype)
    k_ref = np.zeros((batch, kv_heads, sequence, head_dim), dtype=storage_dtype)
    v_ref = x_np.reshape(batch, sequence, kv_heads, value_dim).transpose(0, 2, 1, 3)
    expected_attention = _cast_reference_to_storage_dtype(_attention_reference(q_ref, k_ref, v_ref, scale=scale), dtype)
    expected_merged = _pack_bshd_dense_storage(expected_attention).reshape(batch * sequence, heads * value_dim)

    wrong_dense_bhsd_merge_reference = expected_attention.reshape(batch * sequence, heads * value_dim)
    assert not np.allclose(
        expected_merged.astype(np.float32),
        wrong_dense_bhsd_merge_reference.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    ), "projection-layout sentinel is degenerate; this test would not catch a bad post-attention merge"

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "qw": _host_to_gpu(qw_np, dtype, stream),
        "kw": _host_to_gpu(kw_np, dtype, stream),
        "vw": _host_to_gpu(vw_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [batch * sequence, heads * value_dim]
    stage_kinds = eq._debug_stage_kinds(inputs_gpu)
    assert sum(kind.startswith("Matmul") for kind in stage_kinds) == 3
    assert sum(kind.startswith("Attention") for kind in stage_kinds) == 1
    assert stage_kinds[-1].startswith("Attention")

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected_merged, dtype)

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
    plain_causal_expected = _attention_reference(
        q_np, k_np, v_np, scale=scale, mask_kind=AttentionMaskKind.causal_top_left)
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
def test_attention_paged_kv_cache_forward_decode_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    kv_len = 8
    block_size = 4
    scale = 0.91 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=kv_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_ref, v_ref = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=1, kv_len=kv_len, dtype=dtype)
    k_container = _paged_kv_container(k_ref, block_size)
    v_container = _paged_kv_container(v_ref, block_size)
    q_lengths = np.ones((2,), dtype=np.int32)
    kv_lengths = np.asarray([8, 5], dtype=np.int32)
    expected = _attention_reference(q_np, k_ref, v_ref, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_container, dtype, stream),
        "v": _host_to_gpu(v_container, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "page_table_k": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
        "page_table_v": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, 1, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_paged_kv_cache_forward_gqa_indirect_page_table_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    query_len = 2
    kv_len = 9
    block_size = 4
    page_table = np.asarray([[[[2], [5], [1]]], [[[7], [0], [6]]]], dtype=np.int32)
    scale = 0.83 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=kv_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_ref, v_ref = _attention_inputs(
        batch=2, query_heads=4, kv_heads=2, query_len=query_len, kv_len=kv_len, dtype=dtype)
    k_container = _paged_kv_container_from_page_table(k_ref, block_size, page_table)
    v_container = _paged_kv_container_from_page_table(v_ref, block_size, page_table)
    q_lengths = np.full((2,), query_len, dtype=np.int32)
    kv_lengths = np.asarray([9, 6], dtype=np.int32)
    expected = _attention_reference(q_np, k_ref, v_ref, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_container, dtype, stream),
        "v": _host_to_gpu(v_container, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "page_table_k": _host_to_gpu(page_table, thor.DataType.int32, stream),
        "page_table_v": _host_to_gpu(page_table, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4, query_len, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_paged_kv_cache_forward_mqa_distinct_kv_page_tables_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    query_len = 1
    kv_len = 10
    block_size = 4
    page_table_k_np = np.asarray([[[[4], [1], [6]]], [[[0], [5], [2]]]], dtype=np.int32)
    page_table_v_np = np.asarray([[[[3], [7], [1]]], [[[8], [2], [5]]]], dtype=np.int32)
    scale = 0.79 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=kv_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_ref, v_ref = _attention_inputs(
        batch=2, query_heads=4, kv_heads=1, query_len=query_len, kv_len=kv_len, dtype=dtype)
    k_container = _paged_kv_container_from_page_table(k_ref, block_size, page_table_k_np)
    v_container = _paged_kv_container_from_page_table(v_ref, block_size, page_table_v_np)
    q_lengths = np.full((2,), query_len, dtype=np.int32)
    kv_lengths = np.asarray([10, 7], dtype=np.int32)
    expected = _attention_reference(q_np, k_ref, v_ref, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_container, dtype, stream),
        "v": _host_to_gpu(v_container, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "page_table_k": _host_to_gpu(page_table_k_np, thor.DataType.int32, stream),
        "page_table_v": _host_to_gpu(page_table_v_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4, query_len, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_paged_kv_cache_forward_bf16_decode_matches_reference():
    dtype = thor.DataType.bf16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    kv_len = 8
    block_size = 4
    scale = 0.67 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=kv_len,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_ref, v_ref = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=1, kv_len=kv_len, dtype=dtype)
    k_container = _paged_kv_container(k_ref, block_size)
    v_container = _paged_kv_container(v_ref, block_size)
    q_lengths = np.ones((2,), dtype=np.int32)
    kv_lengths = np.asarray([8, 6], dtype=np.int32)
    expected = _attention_reference(q_np, k_ref, v_ref, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_container, dtype, stream),
        "v": _host_to_gpu(v_container, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "page_table_k": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
        "page_table_v": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, 1, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_paged_kv_cache_forward_sliding_window_bottom_right_decode_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    query_len = 3
    kv_len = 11
    block_size = 4
    scale = 0.73 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=kv_len,
        mask_kind=AttentionMaskKind.sliding_window_bottom_right,
        diagonal_left_bound=3,
        diagonal_right_bound=1,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_ref, v_ref = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=query_len, kv_len=kv_len, dtype=dtype)
    k_container = _paged_kv_container(k_ref, block_size)
    v_container = _paged_kv_container(v_ref, block_size)
    q_lengths = np.full((2,), query_len, dtype=np.int32)
    kv_lengths = np.asarray([11, 8], dtype=np.int32)
    expected = _attention_reference(
        q_np,
        k_ref,
        v_ref,
        scale=scale,
        mask_kind=AttentionMaskKind.sliding_window_bottom_right,
        diagonal_left_bound=3,
        diagonal_right_bound=1,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_container, dtype, stream),
        "v": _host_to_gpu(v_container, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "page_table_k": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
        "page_table_v": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, query_len, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_paged_kv_cache_forward_multitoken_bottom_right_decode_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    query_len = 3
    kv_len = 9
    block_size = 4
    scale = 0.71 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=kv_len,
        mask_kind=AttentionMaskKind.causal_bottom_right,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_ref, v_ref = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=query_len, kv_len=kv_len, dtype=dtype)
    k_container = _paged_kv_container(k_ref, block_size)
    v_container = _paged_kv_container(v_ref, block_size)
    q_lengths = np.full((2,), query_len, dtype=np.int32)
    kv_lengths = np.full((2,), kv_len, dtype=np.int32)
    expected = _attention_reference(
        q_np,
        k_ref,
        v_ref,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_bottom_right,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_container, dtype, stream),
        "v": _host_to_gpu(v_container, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "page_table_k": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
        "page_table_v": _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 2, query_len, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_attention_paged_kv_cache_requires_int32_page_table_shape():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    kv_len = 8
    block_size = 4
    eq = ex.compile(
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            paged_kv_max_sequence_length=kv_len,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ),
        device_num=0,
    )

    q_np, k_ref, v_ref = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=1, kv_len=kv_len, dtype=dtype)
    stream = Stream(gpu_num=0)
    base_inputs = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(_paged_kv_container(k_ref, block_size), dtype, stream),
        "v": _host_to_gpu(_paged_kv_container(v_ref, block_size), dtype, stream),
        "q_seq_len": _host_to_gpu(np.ones((2,), dtype=np.int32), thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(np.asarray([8, 5], dtype=np.int32), thor.DataType.int32, stream),
    }

    bad_dtype = dict(base_inputs)
    bad_dtype["page_table_k"] = _host_to_gpu(
        _page_table(2, kv_len, block_size).astype(np.int64), thor.DataType.int64, stream)
    bad_dtype["page_table_v"] = _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream)
    with pytest.raises(RuntimeError, match="paged KV page_table_k dtype must be INT32"):
        eq.stamp(bad_dtype, stream)

    bad_shape = dict(base_inputs)
    bad_shape["page_table_k"] = _host_to_gpu(np.zeros((2, 1, 3, 1), dtype=np.int32), thor.DataType.int32, stream)
    bad_shape["page_table_v"] = _host_to_gpu(_page_table(2, kv_len, block_size), thor.DataType.int32, stream)
    with pytest.raises(RuntimeError, match=r"paged KV page_table_k shape must be \[B,1,ceil\(Skv/block_k\),1\]"):
        eq.stamp(bad_shape, stream)


def test_attention_paged_kv_cache_rejects_unsupported_combinations_early():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")

    with pytest.raises(RuntimeError, match="page_table_k and page_table_v"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            page_table_k=page_table_k,
            paged_kv_max_sequence_length=64,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="paged KV attention requires paged_kv_max_sequence_length > 0"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="paged KV attention requires q_seq_len and kv_seq_len"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            paged_kv_max_sequence_length=64,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="paged KV attention cannot currently be combined with additive bias"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            bias=bias,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            paged_kv_max_sequence_length=64,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="paged KV attention is inference-only"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            paged_kv_max_sequence_length=64,
            dropout_probability=0.25,
            dropout_seed=dropout_seed,
            dropout_offset=dropout_offset,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="ragged attention and paged KV cache cannot be combined"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            q_ragged_offsets=q_offsets,
            kv_ragged_offsets=kv_offsets,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            paged_kv_max_sequence_length=64,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


def test_attention_paged_kv_cache_backward_rejects_until_training_semantics_are_defined():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    page_table_k = ex.input("page_table_k")
    page_table_v = ex.input("page_table_v")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        paged_kv_max_sequence_length=64,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    with pytest.raises(RuntimeError, match="paged KV cache is not enabled"):
        fwd_eq.compile_backward(["q", "k", "v"])


@pytest.mark.cuda
def test_attention_forward_ragged_offsets_plans_single_stage_and_validates_shape():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    scale = 0.61 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([6, 4], dtype=np.int32)
    kv_lengths = np.asarray([7, 5], dtype=np.int32)
    q_offsets_np = _ragged_element_offsets(q_lengths, heads=2, dim=16)
    kv_offsets_np = _ragged_element_offsets(kv_lengths, heads=2, dim=16)
    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(q_offsets_np, thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(kv_offsets_np, thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 6, 2, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    assert list(stamped.output().dimensions) == [2, 6, 2, 16]


@pytest.mark.cuda
def test_attention_forward_ragged_offsets_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    scale = 0.58 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([4, 2], dtype=np.int32)
    kv_lengths = np.asarray([5, 3], dtype=np.int32)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    expected_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected, dtype), q_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=2, dim=16), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=16), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 6, 2, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_storage = _copy_to_host(stamped.output(), dtype, stream)
    got_valid = _packed_bshd_ragged_valid_values(got_storage, q_lengths).astype(np.float32)
    expected_valid = _packed_bshd_ragged_valid_values(expected_storage, q_lengths).astype(np.float32)
    np.testing.assert_allclose(got_valid, expected_valid, rtol=5e-2, atol=5e-2)


@pytest.mark.cuda
def test_attention_forward_ragged_offsets_gqa_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    scale = 0.61 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=4, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([4, 2], dtype=np.int32)
    kv_lengths = np.asarray([5, 3], dtype=np.int32)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    expected_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected, dtype), q_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=4, dim=16), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=16), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 6, 4, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_storage = _copy_to_host(stamped.output(), dtype, stream)
    _assert_packed_bshd_ragged_close(got_storage, expected_storage, q_lengths)


@pytest.mark.cuda
def test_attention_forward_ragged_offsets_mqa_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    scale = 0.64 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=4, kv_heads=1, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([6, 3], dtype=np.int32)
    kv_lengths = np.asarray([7, 4], dtype=np.int32)
    expected = _attention_reference(q_np, k_np, v_np, scale=scale, q_seq_len=q_lengths, kv_seq_len=kv_lengths)

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    expected_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected, dtype), q_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=4, dim=16), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=1, dim=16), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 6, 4, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_storage = _copy_to_host(stamped.output(), dtype, stream)
    _assert_packed_bshd_ragged_close(got_storage, expected_storage, q_lengths)


@pytest.mark.cuda
def test_attention_forward_ragged_offsets_causal_top_left_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    scale = 0.61 / math.sqrt(16.0)
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([6, 4], dtype=np.int32)
    kv_lengths = np.asarray([7, 5], dtype=np.int32)
    expected = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    expected_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected, dtype), q_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=2, dim=16), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=16), thor.DataType.int32, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 6, 2, 16]
    assert eq._debug_stage_kinds(inputs_gpu) == ["Attention"]
    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_storage = _copy_to_host(stamped.output(), dtype, stream)
    _assert_packed_bshd_ragged_close(got_storage, expected_storage, q_lengths)


@pytest.mark.cuda
def test_attention_ragged_offsets_require_bshd_physical_layouts():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")

    with pytest.raises(RuntimeError, match="Ragged attention requires BSHD physical layouts"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            q_ragged_offsets=q_offsets,
            kv_ragged_offsets=kv_offsets,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_ragged_offsets_require_int32_and_batch_plus_one_shape():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    eq = ex.compile(
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            q_ragged_offsets=q_offsets,
            kv_ragged_offsets=kv_offsets,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ),
        device_num=0,
    )

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([6, 4], dtype=np.int32)
    kv_lengths = np.asarray([7, 5], dtype=np.int32)
    q_storage = _pack_bshd_dense_storage(q_np)
    k_storage = _pack_bshd_dense_storage(k_np)
    v_storage = _pack_bshd_dense_storage(v_np)
    stream = Stream(gpu_num=0)
    base_inputs = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
    }

    bad_dtype = dict(base_inputs)
    bad_dtype["q_offsets"] = _host_to_gpu(
        _ragged_element_offsets(q_lengths, 2, 16).astype(np.int64), thor.DataType.int64, stream)
    bad_dtype["kv_offsets"] = _host_to_gpu(_ragged_element_offsets(kv_lengths, 2, 16), thor.DataType.int32, stream)
    with pytest.raises(RuntimeError, match="ragged q_offsets dtype must be INT32"):
        eq.stamp(bad_dtype, stream)

    bad_shape = dict(base_inputs)
    bad_shape["q_offsets"] = _host_to_gpu(np.asarray([0, 6], dtype=np.int32), thor.DataType.int32, stream)
    bad_shape["kv_offsets"] = _host_to_gpu(_ragged_element_offsets(kv_lengths, 2, 16), thor.DataType.int32, stream)
    with pytest.raises(RuntimeError, match=r"ragged q_offsets shape must be \[B \+ 1\]"):
        eq.stamp(bad_shape, stream)


@pytest.mark.cuda
def test_attention_ragged_offsets_reject_unsupported_combinations_early():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")

    with pytest.raises(RuntimeError, match="q_ragged_offsets and kv_ragged_offsets"):
        ex.scaled_dot_product_attention(
            q, k, v, q_ragged_offsets=q_offsets, output_dtype=dtype, compute_dtype=thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="ragged attention requires q_seq_len and kv_seq_len"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_ragged_offsets=q_offsets,
            kv_ragged_offsets=kv_offsets,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="ragged attention cannot currently be combined"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            bias=bias,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            q_ragged_offsets=q_offsets,
            kv_ragged_offsets=kv_offsets,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )

    with pytest.raises(RuntimeError, match="ragged attention cannot currently be combined"):
        ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            q_ragged_offsets=q_offsets,
            kv_ragged_offsets=kv_offsets,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            dropout_probability=0.25,
            dropout_seed=dropout_seed,
            dropout_offset=dropout_offset,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )


@pytest.mark.cuda
def test_attention_ragged_offsets_feeds_fused_epilogue_and_multi_output_reuses_stage():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    outputs = ex.outputs({
        "attention": attn,
        "shifted": attn * 1.25 + 0.125
    })
    eq = ex.compile(outputs, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=2, query_heads=2, kv_heads=2, query_len=6, kv_len=7, dtype=dtype)
    q_lengths = np.asarray([6, 4], dtype=np.int32)
    kv_lengths = np.asarray([7, 5], dtype=np.int32)
    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, 2, 16), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, 2, 16), thor.DataType.int32, stream),
    }

    assert eq.output_shapes(inputs_gpu) == {
        "attention": [2, 6, 2, 16],
        "shifted": [2, 6, 2, 16]
    }
    kinds = eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1


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
    q_storage = _pack_bshd_dense_storage(q_np)
    k_storage = _pack_bshd_dense_storage(k_np)
    v_storage = _pack_bshd_dense_storage(v_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [1, 4, 2, 16]
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
    return_bias_grad: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    if return_bias_grad:
        if bias is None:
            raise AssertionError("return_bias_grad=True requires additive bias in the reference helper.")
        return dQ, dK, dV, dS

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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_ragged_offsets_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.63 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_lengths = np.asarray([48, 24], dtype=np.int32)
    kv_lengths = np.asarray([55, 31], dtype=np.int32)
    rng = np.random.default_rng(9901)
    dO_np = rng.normal(0.0, 0.25, size=(2, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    dO_storage = _pack_bshd_ragged_storage(dO_np, q_lengths)
    expected_dq_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dq, dtype), q_lengths)
    expected_dk_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dk, dtype), kv_lengths)
    expected_dv_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dv, dtype), kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=2, dim=64), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=64), thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_storage, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [2, 64, 2, 64],
        "k_grad": [2, 64, 2, 64],
        "v_grad": [2, 64, 2, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    got_dq = _copy_to_host(got["q_grad"], dtype, stream)
    got_dk = _copy_to_host(got["k_grad"], dtype, stream)
    got_dv = _copy_to_host(got["v_grad"], dtype, stream)
    np.testing.assert_allclose(
        _packed_bshd_ragged_valid_values(got_dq, q_lengths).astype(np.float32),
        _packed_bshd_ragged_valid_values(expected_dq_storage, q_lengths).astype(np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        _packed_bshd_ragged_valid_values(got_dk, kv_lengths).astype(np.float32),
        _packed_bshd_ragged_valid_values(expected_dk_storage, kv_lengths).astype(np.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        _packed_bshd_ragged_valid_values(got_dv, kv_lengths).astype(np.float32),
        _packed_bshd_ragged_valid_values(expected_dv_storage, kv_lengths).astype(np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_ragged_offsets_gqa_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.67 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=2, query_heads=4, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_lengths = np.asarray([48, 24], dtype=np.int32)
    kv_lengths = np.asarray([55, 31], dtype=np.int32)
    rng = np.random.default_rng(9902)
    dO_np = rng.normal(0.0, 0.25, size=(2, 4, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    dO_storage = _pack_bshd_ragged_storage(dO_np, q_lengths)
    expected_dq_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dq, dtype), q_lengths)
    expected_dk_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dk, dtype), kv_lengths)
    expected_dv_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dv, dtype), kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=4, dim=64), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=64), thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_storage, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [2, 64, 4, 64],
        "k_grad": [2, 64, 2, 64],
        "v_grad": [2, 64, 2, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_packed_bshd_ragged_close(_copy_to_host(got["q_grad"], dtype, stream), expected_dq_storage, q_lengths)
    _assert_packed_bshd_ragged_close(_copy_to_host(got["k_grad"], dtype, stream), expected_dk_storage, kv_lengths)
    _assert_packed_bshd_ragged_close(_copy_to_host(got["v_grad"], dtype, stream), expected_dv_storage, kv_lengths)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_ragged_offsets_causal_top_left_bshd_packed_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.62 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        mask_kind=AttentionMaskKind.causal_top_left,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_lengths = np.asarray([48, 24], dtype=np.int32)
    kv_lengths = np.asarray([55, 31], dtype=np.int32)
    rng = np.random.default_rng(9903)
    dO_np = rng.normal(0.0, 0.25, size=(2, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )

    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)
    dO_storage = _pack_bshd_ragged_storage(dO_np, q_lengths)
    expected_dq_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dq, dtype), q_lengths)
    expected_dk_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dk, dtype), kv_lengths)
    expected_dv_storage = _pack_bshd_ragged_storage(_cast_reference_to_storage_dtype(expected_dv, dtype), kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=2, dim=64), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=64), thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_storage, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [2, 64, 2, 64],
        "k_grad": [2, 64, 2, 64],
        "v_grad": [2, 64, 2, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_packed_bshd_ragged_close(_copy_to_host(got["q_grad"], dtype, stream), expected_dq_storage, q_lengths)
    _assert_packed_bshd_ragged_close(_copy_to_host(got["k_grad"], dtype, stream), expected_dk_storage, kv_lengths)
    _assert_packed_bshd_ragged_close(_copy_to_host(got["v_grad"], dtype, stream), expected_dv_storage, kv_lengths)


@pytest.mark.cuda
def test_attention_backward_with_ragged_offsets_reuses_same_plan_forward_stats_metadata():
    dtype = thor.DataType.fp16
    scale = 0.66 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    q_seq_len = ex.input("q_seq_len")
    kv_seq_len = ex.input("kv_seq_len")
    q_offsets = ex.input("q_offsets")
    kv_offsets = ex.input("kv_offsets")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        q_ragged_offsets=q_offsets,
        kv_ragged_offsets=kv_offsets,
        q_layout=AttentionTensorLayout.bshd,
        k_layout=AttentionTensorLayout.bshd,
        v_layout=AttentionTensorLayout.bshd,
        o_layout=AttentionTensorLayout.bshd,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v"])

    q_np, k_np, v_np = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    q_lengths = np.asarray([48, 24], dtype=np.int32)
    kv_lengths = np.asarray([55, 31], dtype=np.int32)
    q_storage = _pack_bshd_ragged_storage(q_np, q_lengths)
    k_storage = _pack_bshd_ragged_storage(k_np, kv_lengths)
    v_storage = _pack_bshd_ragged_storage(v_np, kv_lengths)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_storage, dtype, stream),
        "k": _host_to_gpu(k_storage, dtype, stream),
        "v": _host_to_gpu(v_storage, dtype, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        "q_offsets": _host_to_gpu(_ragged_element_offsets(q_lengths, heads=2, dim=64), thor.DataType.int32, stream),
        "kv_offsets": _host_to_gpu(_ragged_element_offsets(kv_lengths, heads=2, dim=64), thor.DataType.int32, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [2, 64, 2, 64],
        "k_grad": [2, 64, 2, 64],
        "v_grad": [2, 64, 2, 64],
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    # Stamping is enough to verify that the same-plan cloned forward attention keeps both the
    # ragged-offset metadata and the seq-len metadata required by the retained cuDNN stats path.
    bwd_eq.stamp(inputs_gpu, stream)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np, k_np, v_np, np.float32(2.0) * expected_attn, scale=scale)

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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


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
            rope_base = np.float32(base * (ratio**(float(effective_rotary_dim) / float(effective_rotary_dim - 2))))

    half = effective_rotary_dim // 2
    pair_indices = np.arange(half, dtype=np.float32)
    inv_freq = rope_base**(-2.0 * pair_indices / np.float32(effective_rotary_dim))
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
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream)
    }
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
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream)
    }
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
    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 5, 16]
    }
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
def test_attention_additive_bias_dtype_must_match_compute_dtype_without_hidden_conversion():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    eq = ex.compile(out, device_num=0)

    q_np, k_np, v_np = _attention_inputs(batch=1, query_heads=2, kv_heads=2, query_len=4, kv_len=5, dtype=dtype)
    bias_np = np.zeros((1, 2, 4, 5), dtype=_numpy_storage_dtype(dtype))
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, dtype, stream),
    }

    with pytest.raises(RuntimeError, match="additive bias dtype must match attention compute dtype"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_with_additive_bias_stays_single_attention_backward_stage_and_matches_reference(
):
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
    expected_dq, expected_dk, expected_dv = _attention_backward_reference(
        q_np, k_np, v_np, dO_np, scale=scale, bias=bias_np)

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
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_qkv_and_dbias_with_additive_bias_matches_reference():
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
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v", "bias"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(1888)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv, expected_dbias = _attention_backward_reference(
        q_np, k_np, v_np, dO_np, scale=scale, bias=bias_np, return_bias_grad=True)

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
        "bias_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)
    _assert_close(
        _copy_to_host(got["bias_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_dbias_only_with_additive_bias_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.69 / math.sqrt(64.0)

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
    bwd_eq = fwd_eq.compile_backward(["bias"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["bias_grad"]

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(2888)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    _, _, _, expected_dbias = _attention_backward_reference(
        q_np, k_np, v_np, dO_np, scale=scale, bias=bias_np, return_bias_grad=True)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [1, 2, 64, 64]
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(
        _copy_to_host(got["bias_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_dbias_gqa_with_additive_bias_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.74 / math.sqrt(64.0)

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
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v", "bias"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(3888)
    bias_np = rng.normal(0.0, 0.2, size=(1, 4, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 4, 64, 64)).astype(_numpy_storage_dtype(dtype))
    expected_dq, expected_dk, expected_dv, expected_dbias = _attention_backward_reference(
        q_np, k_np, v_np, dO_np, scale=scale, bias=bias_np, return_bias_grad=True)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 4, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
        "bias_grad": [1, 4, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)
    _assert_close(
        _copy_to_host(got["bias_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_dbias_with_padding_mask_matches_reference_and_zeroes_invalid_positions():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.66 / math.sqrt(64.0)

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
    bwd_eq = fwd_eq.compile_backward(["bias"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(4888)
    bias_np = rng.normal(0.0, 0.2, size=(2, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(2, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    q_lengths = np.asarray([64, 37], dtype=np.int32)
    kv_lengths = np.asarray([59, 41], dtype=np.int32)
    _, _, _, expected_dbias = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
        return_bias_grad=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [2, 2, 64, 64]
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_dbias = _copy_to_host(stamped.outputs()["bias_grad"], dtype, stream)
    _assert_close(got_dbias, _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)
    assert np.count_nonzero(got_dbias[1, :, q_lengths[1]:, :]) == 0
    assert np.count_nonzero(got_dbias[1, :, :, kv_lengths[1]:]) == 0


@pytest.mark.cuda
def test_attention_compile_backward_dbias_with_bf16_inputs_matches_reference():
    dtype = thor.DataType.bf16
    upstream_name = "__grad_output"
    scale = 0.62 / math.sqrt(64.0)

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
    bwd_eq = fwd_eq.compile_backward(["bias"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(5888)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    _, _, _, expected_dbias = _attention_backward_reference(
        q_np, k_np, v_np, dO_np, scale=scale, bias=bias_np, return_bias_grad=True)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [1, 2, 64, 64]
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    _assert_close(
        _copy_to_host(stamped.outputs()["bias_grad"], dtype, stream),
        _cast_reference_to_storage_dtype(expected_dbias, dtype),
        dtype,
    )


@pytest.mark.cuda
def test_attention_same_plan_backward_qkv_and_dbias_with_additive_bias_reuses_forward_stats_and_matches_reference():
    dtype = thor.DataType.fp16
    scale = 0.67 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v", "bias"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(6888)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    expected_attn = _attention_reference(q_np, k_np, v_np, scale=scale, bias=bias_np)
    expected_dq, expected_dk, expected_dv, expected_dbias = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        bias=bias_np,
        return_bias_grad=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
        "bias_grad": [1, 2, 64, 64],
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    # Same-plan backward must reuse the saved Attention stats, but the loss derivative
    # itself is still represented by ordinary fused expression stages. Standalone
    # dBias tests separately assert that public bias_grad does not require an
    # extra dtype-conversion kernel after AttentionBackward.
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)
    _assert_close(
        _copy_to_host(got["bias_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)


@pytest.mark.cuda
def test_attention_same_plan_backward_dbias_only_with_padding_mask_reuses_forward_stats_and_zeroes_invalid_positions():
    dtype = thor.DataType.fp16
    scale = 0.63 / math.sqrt(64.0)

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
    bwd_eq = fwd_eq.compile_backward(["bias"])
    assert bwd_eq.output_names() == ["bias_grad"]

    q_np, k_np, v_np = _attention_inputs(
        batch=2, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(7888)
    bias_np = rng.normal(0.0, 0.2, size=(2, 2, 64, 64)).astype(np.float32)
    q_lengths = np.asarray([64, 39], dtype=np.int32)
    kv_lengths = np.asarray([57, 43], dtype=np.int32)
    expected_attn = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
    )
    _, _, _, expected_dbias = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        bias=bias_np,
        q_seq_len=q_lengths,
        kv_seq_len=kv_lengths,
        return_bias_grad=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "q_seq_len": _host_to_gpu(q_lengths, thor.DataType.int32, stream),
        "kv_seq_len": _host_to_gpu(kv_lengths, thor.DataType.int32, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [2, 2, 64, 64]
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    # Same-plan backward includes fused expression stages for the loss derivative;
    # the training-critical invariant is that attention stats are saved by one
    # Attention stage and consumed by one AttentionBackward stage.
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_dbias = _copy_to_host(stamped.outputs()["bias_grad"], dtype, stream)
    _assert_close(got_dbias, _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)
    assert np.count_nonzero(got_dbias[1, :, q_lengths[1]:, :]) == 0
    assert np.count_nonzero(got_dbias[1, :, :, kv_lengths[1]:]) == 0


@pytest.mark.cuda
@pytest.mark.parametrize(
    "wrt_names,expected_names",
    [
        (["q"], ["q_grad"]),
        (["k"], ["k_grad"]),
        (["v"], ["v_grad"]),
        (["bias"], ["bias_grad"]),
        (["q", "bias"], ["q_grad", "bias_grad"]),
        (["k", "v"], ["k_grad", "v_grad"]),
        (["q", "k", "v", "bias"], ["q_grad", "k_grad", "v_grad", "bias_grad"]),
    ],
)
def test_attention_backward_selector_subsets_with_additive_bias_merge_to_one_cudnn_stage(
        wrt_names: list[str], expected_names: list[str]):
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(wrt_names, error_input_name=upstream_name)
    assert bwd_eq.output_names() == expected_names

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(8188)
    bias_np = rng.normal(0.0, 0.2, size=(1, 4, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 4, 64, 64)).astype(_numpy_storage_dtype(dtype))
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    expected_shapes = {
        "q_grad": [1, 4, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
        "bias_grad": [1, 4, 64, 64],
    }
    assert bwd_eq.output_shapes(inputs_gpu) == {
        name: expected_shapes[name] for name in expected_names
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    assert set(stamped.outputs().keys()) == set(expected_names)


@pytest.mark.cuda
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_attention_same_plan_backward_mqa_gqa_dbias_reuses_forward_stats_and_matches_reference(kv_heads: int):
    dtype = thor.DataType.fp16
    scale = 0.58 / math.sqrt(64.0)
    query_heads = 4

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v", "bias"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=query_heads, kv_heads=kv_heads, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(8288 + kv_heads)
    bias_np = rng.normal(0.0, 0.2, size=(1, query_heads, 64, 64)).astype(np.float32)
    expected_attn = _attention_reference(q_np, k_np, v_np, scale=scale, bias=bias_np)
    expected_dq, expected_dk, expected_dv, expected_dbias = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        bias=bias_np,
        return_bias_grad=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, query_heads, 64, 64],
        "k_grad": [1, kv_heads, 64, 64],
        "v_grad": [1, kv_heads, 64, 64],
        "bias_grad": [1, query_heads, 64, 64],
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    _assert_close(
        _copy_to_host(got["q_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dq, dtype), dtype)
    _assert_close(
        _copy_to_host(got["k_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dk, dtype), dtype)
    _assert_close(
        _copy_to_host(got["v_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dv, dtype), dtype)
    _assert_close(
        _copy_to_host(got["bias_grad"], dtype, stream), _cast_reference_to_storage_dtype(expected_dbias, dtype), dtype)


@pytest.mark.cuda
def test_attention_compile_backward_dbias_with_alibi_causal_mask_matches_reference():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    scale = 0.61 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["bias"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=4, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(8388)
    bias_np = rng.normal(0.0, 0.2, size=(1, 4, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 4, 64, 64)).astype(_numpy_storage_dtype(dtype))
    _, _, _, expected_dbias = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        dO_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        bias=bias_np,
        use_alibi_mask=True,
        return_bias_grad=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [1, 4, 64, 64]
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    _assert_close(
        _copy_to_host(stamped.outputs()["bias_grad"], dtype, stream),
        _cast_reference_to_storage_dtype(expected_dbias, dtype),
        dtype,
    )


@pytest.mark.cuda
def test_attention_compile_backward_dbias_with_dropout_and_additive_bias_stays_single_attention_backward_stage():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    out = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["q", "k", "v", "bias"], error_input_name=upstream_name)

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(8488)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    dO_np = rng.normal(0.0, 0.25, size=(1, 2, 64, 64)).astype(_numpy_storage_dtype(dtype))
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "dropout_seed": _dropout_scalar_gpu(8588, stream),
        "dropout_offset": _dropout_scalar_gpu(8688, stream),
        upstream_name: _host_to_gpu(dO_np, dtype, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "q_grad": [1, 2, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
        "bias_grad": [1, 2, 64, 64],
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["AttentionBackward"]
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    assert got["q_grad"].dimensions == [1, 2, 64, 64]
    assert got["k_grad"].dimensions == [1, 2, 64, 64]
    assert got["v_grad"].dimensions == [1, 2, 64, 64]
    assert got["bias_grad"].dimensions == [1, 2, 64, 64]


@pytest.mark.cuda
def test_attention_same_plan_backward_dbias_with_dropout_and_additive_bias_reuses_forward_stats():
    dtype = thor.DataType.fp16
    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    dropout_seed = ex.input("dropout_seed")
    dropout_offset = ex.input("dropout_offset")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        dropout_probability=0.5,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["bias"])

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=2, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(8788)
    bias_np = rng.normal(0.0, 0.2, size=(1, 2, 64, 64)).astype(np.float32)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
        "dropout_seed": _dropout_scalar_gpu(8888, stream),
        "dropout_offset": _dropout_scalar_gpu(8988, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [1, 2, 64, 64]
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")
    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    assert stamped.outputs()["bias_grad"].dimensions == [1, 2, 64, 64]


@pytest.mark.cuda
@pytest.mark.parametrize(
    "requested_names, expected_names",
    [
        (["q"], ["q_grad"]),
        (["k"], ["k_grad"]),
        (["v"], ["v_grad"]),
        (["bias"], ["bias_grad"]),
        (["q", "bias"], ["q_grad", "bias_grad"]),
        (["k", "v"], ["k_grad", "v_grad"]),
        (["q", "k", "v", "bias"], ["q_grad", "k_grad", "v_grad", "bias_grad"]),
    ],
)
def test_attention_same_plan_backward_selector_subsets_share_saved_stats_attention_backward_stage(
        requested_names: list[str], expected_names: list[str]):
    dtype = thor.DataType.fp16
    scale = 0.55 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    # Use a loss whose derivative depends on the forward attention output.
    # A plain reduce_sum(attn) has constant upstream gradient, so compile_backward
    # can legally use the standalone/rematerialized AttentionBackward path without
    # preserving an explicit Attention stage in the same plan.
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(requested_names)
    assert bwd_eq.output_names() == expected_names

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=2, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(9088 + len(requested_names) * 17)
    bias_np = rng.normal(0.0, 0.2, size=(1, 4, 64, 64)).astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
    }

    expected_shapes = {
        "q_grad": [1, 4, 64, 64],
        "k_grad": [1, 2, 64, 64],
        "v_grad": [1, 2, 64, 64],
        "bias_grad": [1, 4, 64, 64],
    }
    assert bwd_eq.output_shapes(inputs_gpu) == {
        name: expected_shapes[name] for name in expected_names
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = stamped.outputs()
    assert set(got.keys()) == set(expected_names)
    if "bias_grad" in got:
        assert got["bias_grad"].dtype == dtype


@pytest.mark.cuda
def test_attention_same_plan_backward_dbias_with_alibi_causal_mask_reuses_forward_stats_and_matches_reference():
    dtype = thor.DataType.fp16
    scale = 0.57 / math.sqrt(64.0)

    q = ex.input("q")
    k = ex.input("k")
    v = ex.input("v")
    bias = ex.input("bias")
    attn = ex.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask_kind=AttentionMaskKind.causal_top_left,
        use_alibi_mask=True,
        attention_scale=scale,
        output_dtype=dtype,
        compute_dtype=thor.DataType.fp32,
    )
    loss = ex.reduce_sum(attn * attn, axis=[0, 1, 2, 3], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["bias"])
    assert bwd_eq.output_names() == ["bias_grad"]

    q_np, k_np, v_np = _attention_inputs(
        batch=1, query_heads=4, kv_heads=4, query_len=64, kv_len=64, qk_dim=64, v_dim=64, dtype=dtype)
    rng = np.random.default_rng(9188)
    bias_np = rng.normal(0.0, 0.2, size=(1, 4, 64, 64)).astype(np.float32)
    expected_attn = _attention_reference(
        q_np,
        k_np,
        v_np,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        bias=bias_np,
        use_alibi_mask=True,
    )
    _, _, _, expected_dbias = _attention_backward_reference(
        q_np,
        k_np,
        v_np,
        np.float32(2.0) * expected_attn,
        scale=scale,
        mask_kind=AttentionMaskKind.causal_top_left,
        bias=bias_np,
        use_alibi_mask=True,
        return_bias_grad=True,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "q": _host_to_gpu(q_np, dtype, stream),
        "k": _host_to_gpu(k_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, thor.DataType.fp32, stream),
    }

    assert bwd_eq.output_shapes(inputs_gpu) == {
        "bias_grad": [1, 4, 64, 64]
    }
    kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert kinds.count("Attention") == 1
    assert kinds.count("AttentionBackward") == 1
    assert kinds.index("Attention") < kinds.index("AttentionBackward")

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()
    got_dbias_tensor = stamped.outputs()["bias_grad"]
    assert got_dbias_tensor.dtype == dtype
    _assert_close(
        _copy_to_host(got_dbias_tensor, dtype, stream),
        _cast_reference_to_storage_dtype(expected_dbias, dtype),
        dtype,
    )


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
