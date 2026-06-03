from __future__ import annotations

import math
import os
from dataclasses import dataclass
from math import prod
from typing import Callable

import pytest
import thor
from thor.physical import (
    AttentionMaskKind,
    AttentionTensorLayout,
    DeviceType,
    Expression as ex,
    PhysicalTensor,
    Placement,
    RotaryScalingKind,
    Stream,
    cudnn_frontend_attention_available,
)

GPU_NUM = int(os.getenv("THOR_ATTENTION_PERF_GPU", os.getenv("THOR_EXPR_PERF_GPU", "0")))

# Keep the suite practical by default, but make every expensive dimension tunable
# so local profiling can scale up without changing the benchmark source.
WARMUP_ITERS = int(os.getenv("THOR_ATTENTION_PERF_WARMUP_ITERS", "6"))
MEASURE_ITERS = int(os.getenv("THOR_ATTENTION_PERF_MEASURE_ITERS", "32"))
MIN_ROTATING_POOL_BYTES = int(os.getenv("THOR_ATTENTION_PERF_MIN_POOL_BYTES", str(512 * 1024 * 1024)))
MAX_POOL_SLOTS = int(os.getenv("THOR_ATTENTION_PERF_MAX_POOL_SLOTS", "16"))
EXPLICIT_POOL_SLOTS = os.getenv("THOR_ATTENTION_PERF_POOL_SLOTS")
CASE_FILTER = os.getenv("THOR_ATTENTION_PERF_CASE_FILTER")
INITIALIZE_INPUTS = os.getenv("THOR_ATTENTION_PERF_INITIALIZE_INPUTS", "1") != "0"
ENABLE_LARGE_CASES = os.getenv("THOR_ATTENTION_PERF_ENABLE_LARGE_CASES", "0") != "0"
REFERENCE_SEQUENCE = int(os.getenv("THOR_ATTENTION_PERF_REFERENCE_SEQUENCE", "2048"))
LARGE_SEQUENCE = int(os.getenv("THOR_ATTENTION_PERF_LARGE_SEQUENCE", "2048"))

DTYPES = [thor.DataType.fp16, thor.DataType.bf16]


@dataclass(frozen=True)
class AttentionPerfCase:
    name: str
    builder: Callable[[thor.DataType], tuple]
    description: str
    requires_large_opt_in: bool = False


def _dtype_name(dtype: thor.DataType) -> str:
    return str(dtype).split(".")[-1]


def _rotary_scaling_kind_name(value: RotaryScalingKind) -> str:
    return str(value).split(".")[-1]


def _bytes_per_element(dtype: thor.DataType) -> int:
    if dtype == thor.DataType.fp32:
        return 4
    if dtype in (thor.DataType.fp16, thor.DataType.bf16):
        return 2
    if dtype in (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2):
        return 1
    raise AssertionError(f"Unhandled dtype: {dtype}")


def _tensor_bytes(shape: tuple[int, ...], dtype: thor.DataType) -> int:
    return prod(shape) * _bytes_per_element(dtype)


def _input_bytes_per_slot(input_shapes: dict[str, tuple[int, ...]], dtype: thor.DataType) -> int:
    return sum(_tensor_bytes(shape, dtype) for shape in input_shapes.values())


def _choose_pool_slots(input_bytes_per_slot: int) -> int:
    if EXPLICIT_POOL_SLOTS is not None:
        return max(1, int(EXPLICIT_POOL_SLOTS))
    if input_bytes_per_slot <= 0:
        return 2
    slots = math.ceil(MIN_ROTATING_POOL_BYTES / float(input_bytes_per_slot))
    return max(2, min(MAX_POOL_SLOTS, int(slots)))


def _rotation_stride(pool_slots: int) -> int:
    if pool_slots <= 1:
        return 1
    for candidate in (5, 7, 3, 11, 13):
        if math.gcd(candidate, pool_slots) == 1:
            return candidate
    return 1


def _physical_tensor(device_type: DeviceType, shape: tuple[int, ...], dtype: thor.DataType) -> PhysicalTensor:
    device_num = GPU_NUM if device_type == DeviceType.gpu else 0
    return PhysicalTensor(
        Placement(device_type, device_num),
        PhysicalTensor.Descriptor(dtype, list(shape)),
    )


def _gpu_tensor(shape: tuple[int, ...], dtype: thor.DataType) -> PhysicalTensor:
    return _physical_tensor(DeviceType.gpu, shape, dtype)


def _zero_cpu_tensors(input_shapes: dict[str, tuple[int, ...]], dtype: thor.DataType) -> dict[str, PhysicalTensor]:
    if not INITIALIZE_INPUTS:
        return {}

    zeros: dict[str, PhysicalTensor] = {}
    for name, shape in input_shapes.items():
        tensor = _physical_tensor(DeviceType.cpu, shape, dtype)
        tensor.numpy().fill(0)
        zeros[name] = tensor
    return zeros


def _stamp_program(program, inputs: dict[str, PhysicalTensor], stream: Stream):
    # FusedEquation.stamp(inputs, stream) covers the hand-built expression cases.
    # DynamicExpression.stamp(inputs, outputs, stream) covers public API layers, whose
    # physical expression is selected by the C++ layer builder at stamp time.
    if isinstance(program, thor.physical.DynamicExpression):
        return program.stamp(inputs, {}, stream)
    return program.stamp(inputs, stream)


def _unpack_built_case(built: tuple):
    if len(built) == 4:
        program, input_shapes, output_shape, flops_per_launch = built
        return program, input_shapes, output_shape, flops_per_launch, {}
    if len(built) == 5:
        program, input_shapes, output_shape, flops_per_launch, metadata = built
        return program, input_shapes, output_shape, flops_per_launch, dict(metadata)
    raise AssertionError(f"Unexpected benchmark builder return arity: {len(built)}")


def _debug_program_stage_kinds(
    program,
    input_shapes: dict[str, tuple[int, ...]],
    dtype: thor.DataType,
    stream: Stream,
) -> tuple[list[str], list[str]]:
    """Return logical compiled stages and concrete stamped runtime stages for a benchmark case."""
    inputs = {
        name: _gpu_tensor(shape, dtype) for name, shape in input_shapes.items()
    }

    if isinstance(program, thor.physical.DynamicExpression):
        prepared = program.prepare(inputs, {}, stream)
        compiled_stage_kinds = prepared.equation._debug_stage_kinds(
            prepared.stamp_inputs,
            tensor_scalar_inputs=prepared.tensor_scalar_inputs,
        )
        stamped = prepared.stamp()
    else:
        compiled_stage_kinds = program._debug_stage_kinds(inputs)
        stamped = _stamp_program(program, inputs, stream)

    runtime_stage_kinds = stamped._debug_stage_kinds()
    stream.synchronize()
    return list(compiled_stage_kinds), list(runtime_stage_kinds)


def _stage_summary(stage_kinds: list[str]) -> str:
    return " -> ".join(stage_kinds)


def _stage_count(stage_kinds: list[str], prefix: str) -> int:
    return sum(1 for stage in stage_kinds if stage.startswith(prefix))


def _make_stamped_launch_pool(program, input_shapes: dict[str, tuple[int, ...]], dtype: thor.DataType, stream: Stream):
    """
    Build a pool of independently addressed device inputs and stamped programs.

    Reusing a single Q/K/V/token tensor in the timed loop makes attention look
    better than it is because later iterations can hit L2/cache-resident input
    data.  The benchmark therefore rotates across a device-resident input pool
    large enough to exceed normal cache capacity while still keeping allocation,
    stamping, and output materialization outside the timed region.
    """
    input_bytes = _input_bytes_per_slot(input_shapes, dtype)
    pool_slots = _choose_pool_slots(input_bytes)

    zero_inputs = _zero_cpu_tensors(input_shapes, dtype)
    launches: list[Callable[[], None]] = []
    for _ in range(pool_slots):
        inputs = {
            name: _gpu_tensor(shape, dtype) for name, shape in input_shapes.items()
        }
        for name, tensor in inputs.items():
            if name in zero_inputs:
                tensor.copy_from_async(zero_inputs[name], stream)
        stamped = _stamp_program(program, inputs, stream)
        _ = stamped.output()

        def launch(stamped=stamped, inputs=inputs) -> None:
            # Capture inputs as a default argument so the Python tensor owners stay
            # alive for as long as this stamped launch remains in the pool.
            _ = inputs
            stamped.run()

        launches.append(launch)

    stream.synchronize()
    return launches, input_bytes, pool_slots


def _benchmark_rotating_launches(launches: list[Callable[[], None]], stream: Stream) -> float:
    """Return elapsed seconds for MEASURE_ITERS cached launches over a rotating input pool."""
    assert launches
    pool_slots = len(launches)
    stride = _rotation_stride(pool_slots)

    # Trigger JIT/cache population for every stamped plan and pay any first-use
    # runtime setup outside the measured region.
    for launch in launches:
        launch()
    stream.synchronize()

    for launch in launches:
        launch()
    stream.synchronize()

    # Sweep at least the whole pool immediately before timing so the first
    # measured launch is not the same tensor address as the last warm-cache run.
    warmup_launches = max(WARMUP_ITERS, pool_slots)
    slot = 0
    for _ in range(warmup_launches):
        launches[slot]()
        slot = (slot + stride) % pool_slots
    stream.synchronize()

    start = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
    for _ in range(MEASURE_ITERS):
        launches[slot]()
        slot = (slot + stride) % pool_slots
    end = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
    elapsed_ms: float = end.synchronize_and_report_elapsed_time_ms(start)
    return elapsed_ms / 1000.0


def _attention_flops(
    *,
    batch: int,
    query_len: int,
    kv_len: int,
    query_heads: int,
    qk_dim: int,
    v_dim: int,
) -> int:
    # Count Q*K^T and P*V FMAs as two FLOPs each.  This intentionally excludes
    # softmax/mask bookkeeping because cuDNN exposes several implementation
    # choices there and the matmul work dominates real prefill cases.
    qk = 2 * batch * query_heads * query_len * kv_len * qk_dim
    pv = 2 * batch * query_heads * query_len * kv_len * v_dim
    return qk + pv


def _build_sdpa_case(
    *,
    batch: int,
    query_len: int,
    kv_len: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    metadata: dict[str, str] | None = None,
):

    def build(dtype: thor.DataType):
        q = ex.input("q")
        k = ex.input("k")
        v = ex.input("v")
        scale = 1.0 / math.sqrt(float(qk_dim))
        out = ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            mask_kind=mask_kind,
            attention_scale=scale,
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )
        input_shapes = {
            "q": (batch, query_len, query_heads, qk_dim),
            "k": (batch, kv_len, kv_heads, qk_dim),
            "v": (batch, kv_len, kv_heads, v_dim),
        }
        output_shape = (batch, query_len, query_heads, v_dim)
        flops = _attention_flops(
            batch=batch,
            query_len=query_len,
            kv_len=kv_len,
            query_heads=query_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
        )
        case_metadata = {
            "source": "expression_sdpa"
        }
        if metadata:
            case_metadata.update(metadata)
        return ex.compile(
            out, device_num=GPU_NUM), input_shapes, output_shape, flops, case_metadata

    return build


def _build_sdpa_with_rope_case(
    *,
    batch: int,
    sequence: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    rotary_dim: int = 0,
    interleaved: bool = False,
    scaling_kind: RotaryScalingKind = RotaryScalingKind.none,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 0,
    metadata: dict[str, str] | None = None,
):

    def build(dtype: thor.DataType):
        q = ex.input("q")
        k = ex.input("k")
        v = ex.input("v")
        rope_kwargs = {
            "sequence_axis": 1,
            "head_dim_axis": 3,
            "rotary_dim": rotary_dim,
            "interleaved": interleaved,
            "scaling_kind": scaling_kind,
            "scaling_factor": scaling_factor,
            "original_max_position_embeddings": original_max_position_embeddings,
            "output_dtype": dtype,
            "compute_dtype": thor.DataType.fp32,
        }
        q = ex.rope(q, **rope_kwargs)
        k = ex.rope(k, **rope_kwargs)
        out = ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            mask_kind=mask_kind,
            attention_scale=1.0 / math.sqrt(float(qk_dim)),
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )
        input_shapes = {
            "q": (batch, sequence, query_heads, qk_dim),
            "k": (batch, sequence, kv_heads, qk_dim),
            "v": (batch, sequence, kv_heads, v_dim),
        }
        output_shape = (batch, sequence, query_heads, v_dim)
        flops = _attention_flops(
            batch=batch,
            query_len=sequence,
            kv_len=sequence,
            query_heads=query_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
        )
        case_metadata = {
            "source": "expression_sdpa_rope",
            "use_rope": True,
            "rope_rotary_dim": rotary_dim if rotary_dim != 0 else qk_dim,
            "rope_interleaved": interleaved,
            "rope_scaling_kind": _rotary_scaling_kind_name(scaling_kind),
        }
        if metadata:
            case_metadata.update(metadata)
        return ex.compile(out, device_num=GPU_NUM), input_shapes, output_shape, flops, case_metadata

    return build


def _build_packed_qkv_layer_case(
    *,
    batch: int,
    sequence: int,
    input_features: int,
    output_features: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    has_bias: bool,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    use_rope: bool = False,
    rope_rotary_dim: int = 0,
    rope_interleaved: bool = False,
    rope_scaling_kind: RotaryScalingKind = RotaryScalingKind.none,
    rope_scaling_factor: float = 1.0,
    rope_original_max_position_embeddings: int = 0,
    rope_in_place: bool = False,
):

    def build(dtype: thor.DataType):
        feature_input = ex.input("feature_input")
        qkv_weights = ex.input("qkv_weights")
        output_weights = ex.input("output_weights")

        q_width = query_heads * qk_dim
        k_width = kv_heads * qk_dim
        v_width = kv_heads * v_dim
        qkv_width = q_width + k_width + v_width
        merged_width = query_heads * v_dim

        flat = feature_input.reshape([batch * sequence, input_features])
        qkv = ex.matmul(flat, qkv_weights, compute_dtype=thor.DataType.fp32, output_dtype=dtype)
        if has_bias:
            qkv = qkv + ex.input("qkv_bias")

        batch_stride = sequence * qkv_width
        q = qkv.strided_view(
            [batch, sequence, query_heads, qk_dim],
            [batch_stride, qkv_width, qk_dim, 1],
            0,
        ).with_output_dtype(dtype)
        k = qkv.strided_view(
            [batch, sequence, kv_heads, qk_dim],
            [batch_stride, qkv_width, qk_dim, 1],
            q_width,
        ).with_output_dtype(dtype)
        v = qkv.strided_view(
            [batch, sequence, kv_heads, v_dim],
            [batch_stride, qkv_width, v_dim, 1],
            q_width + k_width,
        ).with_output_dtype(dtype)

        if use_rope:
            rope_kwargs = {
                "sequence_axis": 1,
                "head_dim_axis": 3,
                "rotary_dim": rope_rotary_dim,
                "interleaved": rope_interleaved,
                "scaling_kind": rope_scaling_kind,
                "scaling_factor": rope_scaling_factor,
                "original_max_position_embeddings": rope_original_max_position_embeddings,
                "allow_in_place_materialization": rope_in_place,
                "output_dtype": dtype,
                "compute_dtype": thor.DataType.fp32,
            }
            q = ex.rope(q, **rope_kwargs)
            k = ex.rope(k, **rope_kwargs)

        attn = ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            mask_kind=mask_kind,
            attention_scale=1.0 / math.sqrt(float(qk_dim)),
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ).with_output_dtype(dtype)

        merged = attn.reshape([batch * sequence, merged_width])
        out = ex.matmul(merged, output_weights, compute_dtype=thor.DataType.fp32, output_dtype=dtype)
        if has_bias:
            out = out + ex.input("output_bias")
        out = out.reshape([batch, sequence, output_features]).with_output_dtype(dtype)

        input_shapes = {
            "feature_input": (batch, sequence, input_features),
            "qkv_weights": (input_features, qkv_width),
            "output_weights": (merged_width, output_features),
        }
        if has_bias:
            input_shapes["qkv_bias"] = (qkv_width,)
            input_shapes["output_bias"] = (output_features,)

        output_shape = (batch, sequence, output_features)
        projection_flops = 2 * batch * sequence * input_features * qkv_width
        output_projection_flops = 2 * batch * sequence * merged_width * output_features
        flops = projection_flops + output_projection_flops + _attention_flops(
            batch=batch,
            query_len=sequence,
            kv_len=sequence,
            query_heads=query_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
        )
        metadata = {
            "source": "expression",
            "qkv_projection_mode": "packed",
            "has_bias": has_bias,
            "use_rope": use_rope,
            "rope_in_place": rope_in_place,
        }
        if use_rope:
            metadata.update({
                "rope_rotary_dim": rope_rotary_dim if rope_rotary_dim != 0 else qk_dim,
                "rope_interleaved": rope_interleaved,
                "rope_scaling_kind": _rotary_scaling_kind_name(rope_scaling_kind),
                "rope_in_place": rope_in_place,
            })
        return ex.compile(out, device_num=GPU_NUM), input_shapes, output_shape, flops, metadata

    return build


def _build_split_qkv_layer_case(
    *,
    batch: int,
    sequence: int,
    input_features: int,
    output_features: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    has_bias: bool,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    use_rope: bool = False,
    rope_rotary_dim: int = 0,
    rope_interleaved: bool = False,
    rope_scaling_kind: RotaryScalingKind = RotaryScalingKind.none,
    rope_scaling_factor: float = 1.0,
    rope_original_max_position_embeddings: int = 0,
    rope_in_place: bool = False,
):

    def build(dtype: thor.DataType):
        feature_input = ex.input("feature_input")
        query_weights = ex.input("query_weights")
        key_weights = ex.input("key_weights")
        value_weights = ex.input("value_weights")
        output_weights = ex.input("output_weights")

        q_width = query_heads * qk_dim
        k_width = kv_heads * qk_dim
        v_width = kv_heads * v_dim
        merged_width = query_heads * v_dim

        flat = feature_input.reshape([batch * sequence, input_features])
        q = ex.matmul(flat, query_weights, compute_dtype=thor.DataType.fp32, output_dtype=dtype)
        k = ex.matmul(flat, key_weights, compute_dtype=thor.DataType.fp32, output_dtype=dtype)
        v = ex.matmul(flat, value_weights, compute_dtype=thor.DataType.fp32, output_dtype=dtype)
        if has_bias:
            q = q + ex.input("query_bias")
            k = k + ex.input("key_bias")
            v = v + ex.input("value_bias")

        q = q.reshape([batch, sequence, query_heads, qk_dim]).with_output_dtype(dtype)
        k = k.reshape([batch, sequence, kv_heads, qk_dim]).with_output_dtype(dtype)
        v = v.reshape([batch, sequence, kv_heads, v_dim]).with_output_dtype(dtype)

        if use_rope:
            rope_kwargs = {
                "sequence_axis": 1,
                "head_dim_axis": 3,
                "rotary_dim": rope_rotary_dim,
                "interleaved": rope_interleaved,
                "scaling_kind": rope_scaling_kind,
                "scaling_factor": rope_scaling_factor,
                "original_max_position_embeddings": rope_original_max_position_embeddings,
                "allow_in_place_materialization": rope_in_place,
                "output_dtype": dtype,
                "compute_dtype": thor.DataType.fp32,
            }
            q = ex.rope(q, **rope_kwargs)
            k = ex.rope(k, **rope_kwargs)

        attn = ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            mask_kind=mask_kind,
            attention_scale=1.0 / math.sqrt(float(qk_dim)),
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        ).with_output_dtype(dtype)

        merged = attn.reshape([batch * sequence, merged_width])
        out = ex.matmul(merged, output_weights, compute_dtype=thor.DataType.fp32, output_dtype=dtype)
        if has_bias:
            out = out + ex.input("output_bias")
        out = out.reshape([batch, sequence, output_features]).with_output_dtype(dtype)

        input_shapes = {
            "feature_input": (batch, sequence, input_features),
            "query_weights": (input_features, q_width),
            "key_weights": (input_features, k_width),
            "value_weights": (input_features, v_width),
            "output_weights": (merged_width, output_features),
        }
        if has_bias:
            input_shapes["query_bias"] = (q_width,)
            input_shapes["key_bias"] = (k_width,)
            input_shapes["value_bias"] = (v_width,)
            input_shapes["output_bias"] = (output_features,)

        output_shape = (batch, sequence, output_features)
        projection_flops = 2 * batch * sequence * input_features * (q_width + k_width + v_width)
        output_projection_flops = 2 * batch * sequence * merged_width * output_features
        flops = projection_flops + output_projection_flops + _attention_flops(
            batch=batch,
            query_len=sequence,
            kv_len=sequence,
            query_heads=query_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
        )
        metadata = {
            "source": "expression",
            "qkv_projection_mode": "split",
            "use_rope": use_rope,
            "rope_in_place": rope_in_place,
        }
        if use_rope:
            metadata.update({
                "rope_rotary_dim": rope_rotary_dim if rope_rotary_dim != 0 else qk_dim,
                "rope_interleaved": rope_interleaved,
                "rope_scaling_kind": _rotary_scaling_kind_name(rope_scaling_kind),
                "rope_in_place": rope_in_place,
            })
        return ex.compile(out, device_num=GPU_NUM), input_shapes, output_shape, flops, metadata

    return build


def _build_public_attention_layer_case(
    *,
    batch: int,
    sequence: int,
    input_features: int,
    output_features: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    has_bias: bool,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
    use_rope: bool = False,
    rope_rotary_dim: int = 0,
    rope_interleaved: bool = False,
    rope_scaling_kind: RotaryScalingKind = RotaryScalingKind.none,
    rope_scaling_factor: float = 1.0,
    rope_original_max_position_embeddings: int = 0,
    rope_in_place: bool = False,
):

    def build(dtype: thor.DataType):
        if not hasattr(thor.layers, "Attention"):
            pytest.skip("thor.layers.Attention is not exposed in this build")

        effective_rope_rotary_dim = rope_rotary_dim if rope_rotary_dim != 0 else qk_dim

        if mask_kind == AttentionMaskKind.none:
            mask_kind_name = "none"
        elif mask_kind == AttentionMaskKind.causal_top_left:
            mask_kind_name = "causal_top_left"
        elif mask_kind == AttentionMaskKind.causal_bottom_right:
            mask_kind_name = "causal_bottom_right"
        elif mask_kind == AttentionMaskKind.sliding_window_top_left:
            mask_kind_name = "sliding_window_top_left"
        elif mask_kind == AttentionMaskKind.sliding_window_bottom_right:
            mask_kind_name = "sliding_window_bottom_right"
        else:
            raise AssertionError(f"Unhandled AttentionMaskKind: {mask_kind}")

        network = thor.Network(
            f"attention_perf_public_{_dtype_name(dtype)}_{batch}_{sequence}_{input_features}_{output_features}_"
            f"{query_heads}_{kv_heads}_{qk_dim}_{v_dim}")
        # Public API shape is per-example [sequence, features].  The layer's DynamicExpression
        # sees the physical batched tensor [batch, sequence, features] when stamped.
        feature_input = thor.layers.NetworkInput(network, "feature_input", [sequence, input_features],
                                                 dtype).get_feature_output()
        layer = thor.layers.Attention(
            network,
            feature_input,
            num_heads=query_heads,
            num_key_value_heads=kv_heads,
            head_dim=qk_dim,
            value_dim=v_dim,
            output_features=output_features,
            has_bias=has_bias,
            mask_kind=mask_kind_name,
            attention_scale=1.0 / math.sqrt(float(qk_dim)),
            use_rope=use_rope,
            rope_rotary_dim=rope_rotary_dim,
            rope_interleaved=rope_interleaved,
            rope_scaling_kind=_rotary_scaling_kind_name(rope_scaling_kind),
            rope_scaling_factor=rope_scaling_factor,
            rope_original_max_position_embeddings=rope_original_max_position_embeddings,
            rope_in_place=rope_in_place,
            weights_data_type=dtype,
            compute_data_type=thor.DataType.fp32,
            output_data_type=dtype,
        )

        q_width = query_heads * qk_dim
        k_width = kv_heads * qk_dim
        v_width = kv_heads * v_dim
        qkv_width = q_width + k_width + v_width
        merged_width = query_heads * v_dim

        input_shapes = {
            "feature_input": (batch, sequence, input_features)
        }
        projection_mode = layer._debug_qkv_projection_mode()
        if projection_mode == "packed":
            input_shapes["qkv_weights"] = (input_features, qkv_width)
            if has_bias:
                input_shapes["qkv_bias"] = (qkv_width,)
        elif projection_mode == "split":
            input_shapes["query_weights"] = (input_features, q_width)
            input_shapes["key_weights"] = (input_features, k_width)
            input_shapes["value_weights"] = (input_features, v_width)
            if has_bias:
                input_shapes["query_bias"] = (q_width,)
                input_shapes["key_bias"] = (k_width,)
                input_shapes["value_bias"] = (v_width,)
        else:
            raise AssertionError(f"Unexpected public Attention projection mode: {projection_mode}")
        input_shapes["output_weights"] = (merged_width, output_features)
        if has_bias:
            input_shapes["output_bias"] = (output_features,)

        output_shape = (batch, sequence, output_features)
        projection_flops = 2 * batch * sequence * input_features * qkv_width
        output_projection_flops = 2 * batch * sequence * merged_width * output_features
        flops = projection_flops + output_projection_flops + _attention_flops(
            batch=batch,
            query_len=sequence,
            kv_len=sequence,
            query_heads=query_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
        )
        metadata = {
            "source": "thor.layers.Attention",
            "qkv_projection_mode": projection_mode,
            "has_bias": has_bias,
            "use_rope": use_rope,
            "rope_in_place": rope_in_place,
        }
        if use_rope:
            metadata.update({
                "rope_rotary_dim": effective_rope_rotary_dim,
                "rope_interleaved": rope_interleaved,
                "rope_scaling_kind": _rotary_scaling_kind_name(rope_scaling_kind),
                "rope_in_place": rope_in_place,
            })
        return layer._debug_expression(), input_shapes, output_shape, flops, metadata

    return build


def _build_packed_qkv_attention_backward_case(
    *,
    batch: int,
    sequence: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    mask_kind: AttentionMaskKind = AttentionMaskKind.none,
):

    def build(dtype: thor.DataType):
        q_width = query_heads * qk_dim
        k_width = kv_heads * qk_dim
        v_width = kv_heads * v_dim
        qkv_width = q_width + k_width + v_width

        qkv = ex.input("qkv")
        batch_stride = sequence * qkv_width
        q = qkv.strided_view(
            [batch, sequence, query_heads, qk_dim],
            [batch_stride, qkv_width, qk_dim, 1],
            0,
        ).with_output_dtype(dtype)
        k = qkv.strided_view(
            [batch, sequence, kv_heads, qk_dim],
            [batch_stride, qkv_width, qk_dim, 1],
            q_width,
        ).with_output_dtype(dtype)
        v = qkv.strided_view(
            [batch, sequence, kv_heads, v_dim],
            [batch_stride, qkv_width, v_dim, 1],
            q_width + k_width,
        ).with_output_dtype(dtype)

        out = ex.scaled_dot_product_attention(
            q,
            k,
            v,
            q_layout=AttentionTensorLayout.bshd,
            k_layout=AttentionTensorLayout.bshd,
            v_layout=AttentionTensorLayout.bshd,
            o_layout=AttentionTensorLayout.bshd,
            mask_kind=mask_kind,
            attention_scale=1.0 / math.sqrt(float(qk_dim)),
            output_dtype=dtype,
            compute_dtype=thor.DataType.fp32,
        )
        fwd = ex.compile(out, device_num=GPU_NUM)
        bwd = fwd.compile_backward(["qkv"], error_input_name="__grad_output")
        input_shapes = {
            "qkv": (batch * sequence, qkv_width),
            "__grad_output": (batch, sequence, query_heads, v_dim),
        }
        output_shape = (batch * sequence, qkv_width)
        flops = 2 * _attention_flops(
            batch=batch,
            query_len=sequence,
            kv_len=sequence,
            query_heads=query_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
        )
        return bwd, input_shapes, output_shape, flops

    return build


def _public_attention_case(
    *,
    name: str,
    description: str,
    batch: int,
    sequence: int,
    model_width: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    value_dim: int | None = None,
    has_bias: bool = False,
    mask_kind: AttentionMaskKind = AttentionMaskKind.causal_top_left,
    use_rope: bool = False,
    requires_large_opt_in: bool = False,
) -> AttentionPerfCase:
    value_dim = head_dim if value_dim is None else value_dim
    return AttentionPerfCase(
        name=name,
        builder=_build_public_attention_layer_case(
            batch=batch,
            sequence=sequence,
            input_features=model_width,
            output_features=model_width,
            query_heads=query_heads,
            kv_heads=kv_heads,
            qk_dim=head_dim,
            v_dim=value_dim,
            has_bias=has_bias,
            mask_kind=mask_kind,
            use_rope=use_rope,
        ),
        description=description,
        requires_large_opt_in=requires_large_opt_in,
    )


def _sdpa_reference_case(
    *,
    name: str,
    description: str,
    batch: int,
    sequence: int,
    query_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    mask_kind: AttentionMaskKind,
    reference_shape: str,
    requires_large_opt_in: bool = False,
) -> AttentionPerfCase:
    return AttentionPerfCase(
        name=name,
        builder=_build_sdpa_case(
            batch=batch,
            query_len=sequence,
            kv_len=sequence,
            query_heads=query_heads,
            kv_heads=kv_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            mask_kind=mask_kind,
            metadata={
                "reference_shape": reference_shape
            },
        ),
        description=description,
        requires_large_opt_in=requires_large_opt_in,
    )


CASES = [
    AttentionPerfCase(
        name="sdpa_llama_style_prefill_gqa_s2048_h32_kv8_d128",
        builder=_build_sdpa_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_PREFILL_BATCH", "1")),
            query_len=int(os.getenv("THOR_ATTENTION_PERF_PREFILL_SEQUENCE", "2048")),
            kv_len=int(os.getenv("THOR_ATTENTION_PERF_PREFILL_SEQUENCE", "2048")),
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            mask_kind=AttentionMaskKind.causal_top_left,
            metadata={
                "reference_shape": "thor_local_llama_style_h32_gqa"
            },
        ),
        description="Llama-style causal prefill SDPA with grouped-query attention.",
    ),
    AttentionPerfCase(
        name="sdpa_rope_llama_style_prefill_gqa_s2048_h32_kv8_d128",
        builder=_build_sdpa_with_rope_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_PREFILL_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_PREFILL_SEQUENCE", "2048")),
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            mask_kind=AttentionMaskKind.causal_top_left,
            rotary_dim=128,
            metadata={
                "reference_shape": "thor_local_llama_style_h32_gqa",
                "estimated_tflops_note": "excludes_rope_trig",
            },
        ),
        description="Llama-style causal prefill SDPA with RoPE applied to Q and K before cuDNN attention.",
    ),
    AttentionPerfCase(
        name="sdpa_llama_style_decode_gqa_q1_kv4096_h32_kv8_d128",
        builder=_build_sdpa_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_DECODE_BATCH", "1")),
            query_len=1,
            kv_len=int(os.getenv("THOR_ATTENTION_PERF_DECODE_KV_SEQUENCE", "4096")),
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            mask_kind=AttentionMaskKind.none,
            metadata={
                "reference_shape": "thor_local_llama_style_decode_h32_gqa"
            },
        ),
        description="Single-token decode SDPA over a real KV-cache length.",
    ),
    AttentionPerfCase(
        name="sdpa_bert_style_training_b8_s512_h16_d64",
        builder=_build_sdpa_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_TRAIN_BATCH", "8")),
            query_len=int(os.getenv("THOR_ATTENTION_PERF_TRAIN_SEQUENCE", "512")),
            kv_len=int(os.getenv("THOR_ATTENTION_PERF_TRAIN_SEQUENCE", "512")),
            query_heads=16,
            kv_heads=16,
            qk_dim=64,
            v_dim=64,
            mask_kind=AttentionMaskKind.none,
            metadata={
                "reference_shape": "thor_local_bert_style_mha"
            },
        ),
        description="BERT/encoder-style full MHA training attention.",
    ),
    _sdpa_reference_case(
        name="sdpa_cudnn_frontend_llama31_causal_b1_s2048_h64_kv8_d128",
        batch=1,
        sequence=REFERENCE_SEQUENCE,
        query_heads=64,
        kv_heads=8,
        qk_dim=128,
        v_dim=128,
        mask_kind=AttentionMaskKind.causal_top_left,
        reference_shape="cudnn_frontend_llama31_causal_h64_kv8_d128",
        description="cuDNN Frontend public-reference Llama 3.1 causal SDPA shape.",
    ),
    _sdpa_reference_case(
        name="sdpa_cudnn_frontend_llama31_no_mask_b1_s2048_h64_kv8_d128",
        batch=1,
        sequence=REFERENCE_SEQUENCE,
        query_heads=64,
        kv_heads=8,
        qk_dim=128,
        v_dim=128,
        mask_kind=AttentionMaskKind.none,
        reference_shape="cudnn_frontend_llama31_no_mask_h64_kv8_d128",
        description="cuDNN Frontend public-reference Llama 3.1 non-causal SDPA shape.",
    ),
    _sdpa_reference_case(
        name="sdpa_cudnn_frontend_gpt_oss_causal_b1_s2048_h64_kv8_d64",
        batch=1,
        sequence=REFERENCE_SEQUENCE,
        query_heads=64,
        kv_heads=8,
        qk_dim=64,
        v_dim=64,
        mask_kind=AttentionMaskKind.causal_top_left,
        reference_shape="cudnn_frontend_gpt_oss_causal_h64_kv8_d64",
        description="cuDNN Frontend public-reference GPT-OSS causal SDPA shape.",
    ),
    _sdpa_reference_case(
        name="sdpa_cudnn_frontend_deepseek_v3_causal_b1_s2048_h128_kv128_qk192_v128",
        batch=1,
        sequence=REFERENCE_SEQUENCE,
        query_heads=128,
        kv_heads=128,
        qk_dim=192,
        v_dim=128,
        mask_kind=AttentionMaskKind.causal_top_left,
        reference_shape="cudnn_frontend_deepseek_v3_causal_h128_kv128_qk192_v128",
        description="cuDNN Frontend public-reference DeepSeek V3 causal SDPA shape.",
    ),
    AttentionPerfCase(
        name="expression_packed_qkv_attention_layer_llama_style_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_packed_qkv_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
        ),
        description="Hand-built expression hot path: packed QKV projection, cuDNN SDPA, output projection.",
    ),
    AttentionPerfCase(
        name="expression_split_qkv_attention_layer_llama_style_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_split_qkv_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
        ),
        description="Hand-built expression hot path: split Q/K/V projections, cuDNN SDPA, output projection.",
    ),
    AttentionPerfCase(
        name="expression_split_qkv_rope_attention_layer_llama_style_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_split_qkv_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
            use_rope=True,
            rope_rotary_dim=128,
        ),
        description="Hand-built expression hot path with split Q/K/V projections, RoPE on Q/K, cuDNN SDPA, and output projection.",
    ),
    AttentionPerfCase(
        name="expression_split_qkv_rope_in_place_attention_layer_llama_style_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_split_qkv_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
            use_rope=True,
            rope_rotary_dim=128,
            rope_in_place=True,
        ),
        description="Hand-built expression hot path with split Q/K/V projections and memory-saving in-place RoPE on Q/K.",
    ),
    AttentionPerfCase(
        name="public_attention_layer_llama_style_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_public_attention_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
        ),
        description="Public thor.layers.Attention hot path using the C++ layer's configured QKV projection mode.",
    ),
    AttentionPerfCase(
        name="public_attention_layer_llama_style_rope_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_public_attention_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
            use_rope=True,
        ),
        description="Public thor.layers.Attention hot path with the layer's built-in RoPE option enabled.",
    ),
    AttentionPerfCase(
        name="public_attention_layer_llama_style_rope_in_place_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_public_attention_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=False,
            mask_kind=AttentionMaskKind.causal_top_left,
            use_rope=True,
            rope_in_place=True,
        ),
        description="Public thor.layers.Attention hot path with memory-saving in-place RoPE enabled.",
    ),
    AttentionPerfCase(
        name="public_attention_layer_llama_style_with_bias_b1_s2048_model4096_h32_kv8_d128",
        builder=_build_public_attention_layer_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            input_features=4096,
            output_features=4096,
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            has_bias=True,
            mask_kind=AttentionMaskKind.causal_top_left,
        ),
        description=
        "Public thor.layers.Attention hot path with Q/K/V/output projection biases fused through cuBLASLt epilogues.",
    ),
    _public_attention_case(
        name="public_attention_layer_llama31_70b_style_b1_s2048_model8192_h64_kv8_d128",
        batch=1,
        sequence=LARGE_SEQUENCE,
        model_width=8192,
        query_heads=64,
        kv_heads=8,
        head_dim=128,
        has_bias=False,
        mask_kind=AttentionMaskKind.causal_top_left,
        description="Public Attention layer with Llama-3.1-70B-style hidden width and GQA head layout.",
        requires_large_opt_in=False,  #True,
    ),
    _public_attention_case(
        name="public_attention_layer_gpt3_175b_style_b1_s2048_model12288_h96_kv96_d128",
        batch=1,
        sequence=LARGE_SEQUENCE,
        model_width=12288,
        query_heads=96,
        kv_heads=96,
        head_dim=128,
        has_bias=False,
        mask_kind=AttentionMaskKind.causal_top_left,
        description="Public Attention layer with GPT-3-175B-style dense MHA width/head geometry.",
        requires_large_opt_in=False,  #True,
    ),
    _public_attention_case(
        name="public_attention_layer_frontier_gqa_style_b1_s2048_model12288_h96_kv8_d128",
        batch=1,
        sequence=LARGE_SEQUENCE,
        model_width=12288,
        query_heads=96,
        kv_heads=8,
        head_dim=128,
        has_bias=False,
        mask_kind=AttentionMaskKind.causal_top_left,
        description=(
            "Public Attention layer with a large chat-model-scale hidden width and modern GQA layout; "
            "this is an approximation, not a disclosed ChatGPT/Claude architecture."),
        requires_large_opt_in=False,  #True,
    ),
]

if CASE_FILTER:
    CASES = [case for case in CASES if CASE_FILTER in case.name]
elif not ENABLE_LARGE_CASES:
    CASES = [case for case in CASES if not case.requires_large_opt_in]


@pytest.mark.cuda
@pytest.mark.performance
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_name)
def test_packed_qkv_attention_backward_runtime_has_no_pack_scatter_kernel(dtype: thor.DataType, record_property):
    if not cudnn_frontend_attention_available():
        pytest.skip("Thor was not built with cuDNN Frontend attention support")

    case = AttentionPerfCase(
        name="packed_qkv_attention_backward_direct_cudnn_output_guard",
        builder=_build_packed_qkv_attention_backward_case(
            batch=int(os.getenv("THOR_ATTENTION_PERF_LAYER_BATCH", "1")),
            sequence=int(os.getenv("THOR_ATTENTION_PERF_LAYER_SEQUENCE", "2048")),
            query_heads=32,
            kv_heads=8,
            qk_dim=128,
            v_dim=128,
            mask_kind=AttentionMaskKind.causal_top_left,
        ),
        description="Packed-QKV SDPA backward must write dQ/dK/dV directly into the packed dQKV buffer.",
    )

    stream = Stream(Placement(DeviceType.gpu, GPU_NUM))
    program, input_shapes, output_shape, _, _ = _unpack_built_case(case.builder(dtype))
    inputs = {
        name: _gpu_tensor(shape, dtype) for name, shape in input_shapes.items()
    }

    # DynamicExpression does not expose _debug_stage_kinds directly.  Prepare it first,
    # then inspect the underlying FusedEquation with the exact stamp-time bindings chosen
    # by the public layer builder.  The stamped runtime debug path intentionally reports
    # only concrete runtime stage kinds, so check both surfaces.
    if isinstance(program, thor.physical.DynamicExpression):
        prepared = program.prepare(inputs, {}, stream)
        compiled_stage_kinds = prepared.equation._debug_stage_kinds(
            prepared.stamp_inputs,
            tensor_scalar_inputs=prepared.tensor_scalar_inputs,
        )
        stamped = prepared.stamp()
    else:
        compiled_stage_kinds = program._debug_stage_kinds(inputs)
        stamped = _stamp_program(program, inputs, stream)

    compiled_matmul_stage_kinds = [stage for stage in compiled_stage_kinds if stage.startswith("Matmul")]
    runtime_stage_kinds = stamped._debug_stage_kinds()

    record_property("case", case.name)
    record_property("description", case.description)
    record_property("dtype", _dtype_name(dtype))
    record_property("compiled_stage_kinds", str(compiled_stage_kinds))
    record_property("runtime_stage_kinds", str(runtime_stage_kinds))
    record_property("output_shape", str(output_shape))

    # The logical compiled graph may still contain a FusedKernel pack/scatter stage,
    # but the benchmark hot path must not execute it. The stamped plan is the
    # concrete runtime plan that the benchmark repeatedly launches.
    assert compiled_stage_kinds.count("AttentionBackward") == 1
    assert compiled_stage_kinds.count("FusedKernel") == 1
    assert runtime_stage_kinds == ["AttentionBackward"]
    assert "FusedKernel" not in runtime_stage_kinds


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_name)
def test_public_attention_projection_biases_compile_as_gemm_epilogues(dtype: thor.DataType, record_property):
    if not cudnn_frontend_attention_available():
        pytest.skip("Thor was not built with cuDNN Frontend attention support")
    if not hasattr(thor.layers, "Attention"):
        pytest.skip("thor.layers.Attention is not exposed in this build")

    case = AttentionPerfCase(
        name="public_attention_bias_epilogue_contract",
        builder=_build_public_attention_layer_case(
            batch=2,
            sequence=16,
            input_features=64,
            output_features=64,
            query_heads=4,
            kv_heads=2,
            qk_dim=16,
            v_dim=16,
            has_bias=True,
            mask_kind=AttentionMaskKind.causal_top_left,
        ),
        description="Public Attention projection biases should be rank-1 GEMM addends handled by cuBLASLt epilogues.",
    )

    stream = Stream(Placement(DeviceType.gpu, GPU_NUM))
    program, input_shapes, output_shape, _, metadata = _unpack_built_case(case.builder(dtype))
    inputs = {
        name: _gpu_tensor(shape, dtype) for name, shape in input_shapes.items()
    }

    # DynamicExpression does not expose _debug_stage_kinds directly. Prepare it first,
    # then inspect the underlying FusedEquation with the exact stamp-time bindings chosen
    # by the public layer builder. The stamped runtime debug path intentionally reports
    # only concrete runtime stage kinds, so check both surfaces.
    if isinstance(program, thor.physical.DynamicExpression):
        prepared = program.prepare(inputs, {}, stream)
        compiled_stage_kinds = prepared.equation._debug_stage_kinds(
            prepared.stamp_inputs,
            tensor_scalar_inputs=prepared.tensor_scalar_inputs,
        )
        stamped = prepared.stamp()
    else:
        compiled_stage_kinds = program._debug_stage_kinds(inputs)
        stamped = _stamp_program(program, inputs, stream)

    runtime_stage_kinds = stamped._debug_stage_kinds()
    compiled_matmul_stage_kinds = [stage for stage in compiled_stage_kinds if stage.startswith("Matmul")]
    runtime_matmul_stage_kinds = [stage for stage in runtime_stage_kinds if stage.startswith("Matmul")]

    expected_matmul_stages = 2 if metadata.get("qkv_projection_mode") == "packed" else 4
    record_property("compiled_stage_kinds", str(compiled_stage_kinds))
    record_property("runtime_stage_kinds", str(runtime_stage_kinds))
    record_property("projection_mode", str(metadata.get("qkv_projection_mode")))
    record_property("output_shape", str(output_shape))

    assert len(compiled_matmul_stage_kinds) == expected_matmul_stages
    # opName(ExprOp::GEMM) is exposed as uppercase "GEMM" in the C++ debug string.
    # Accept the older lowercase spelling too so this contract remains robust if the
    # debug helper is normalized later.
    assert all(("op=GEMM" in stage or "op=gemm" in stage) for stage in compiled_matmul_stage_kinds)
    assert len(runtime_matmul_stage_kinds) == expected_matmul_stages
    assert any(stage.startswith("Attention") for stage in runtime_stage_kinds)
    if metadata.get("qkv_projection_mode") == "split":
        assert not any(stage.startswith("FusedKernel") for stage in runtime_stage_kinds)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_name)
def test_rope_qk_materialization_policy_controls_in_place_stage(dtype: thor.DataType, record_property):
    if not cudnn_frontend_attention_available():
        pytest.skip("Thor was not built with cuDNN Frontend attention support")

    stream = Stream(Placement(DeviceType.gpu, GPU_NUM))

    cases = [
        AttentionPerfCase(
            name="sdpa_rope_gqa_grouped_materialization_contract",
            builder=_build_sdpa_with_rope_case(
                batch=2,
                sequence=8,
                query_heads=4,
                kv_heads=2,
                qk_dim=16,
                v_dim=16,
                mask_kind=AttentionMaskKind.causal_top_left,
                rotary_dim=16,
            ),
            description="Standalone SDPA should materialize Q/K RoPE with one grouped fused stage before attention.",
        ),
        AttentionPerfCase(
            name="public_attention_split_qkv_rope_grouped_materialization_contract",
            builder=_build_public_attention_layer_case(
                batch=2,
                sequence=8,
                input_features=64,
                output_features=64,
                query_heads=4,
                kv_heads=2,
                qk_dim=16,
                v_dim=16,
                has_bias=False,
                mask_kind=AttentionMaskKind.causal_top_left,
                use_rope=True,
                rope_rotary_dim=16,
            ),
            description="Public split-QKV Attention defaults to the faster out-of-place grouped RoPE materialization.",
        ),
        AttentionPerfCase(
            name="public_attention_split_qkv_rope_in_place_materialization_contract",
            builder=_build_public_attention_layer_case(
                batch=2,
                sequence=8,
                input_features=64,
                output_features=64,
                query_heads=4,
                kv_heads=2,
                qk_dim=16,
                v_dim=16,
                has_bias=False,
                mask_kind=AttentionMaskKind.causal_top_left,
                use_rope=True,
                rope_rotary_dim=16,
                rope_in_place=True,
            ),
            description="Public split-QKV Attention can opt into memory-saving in-place Q/K RoPE postprocessing.",
        ),
    ]

    for case in cases:
        program, input_shapes, output_shape, _, metadata = _unpack_built_case(case.builder(dtype))
        compiled_stage_kinds, runtime_stage_kinds = _debug_program_stage_kinds(program, input_shapes, dtype, stream)

        record_property(f"{case.name}_compiled_stage_kinds", str(compiled_stage_kinds))
        record_property(f"{case.name}_runtime_stage_kinds", str(runtime_stage_kinds))
        record_property(f"{case.name}_output_shape", str(output_shape))
        record_property(f"{case.name}_metadata", str(metadata))

        assert _stage_count(runtime_stage_kinds, "Attention") == 1
        if metadata.get("rope_in_place"):
            assert _stage_count(runtime_stage_kinds, "InPlaceRope") == 1
            assert _stage_count(compiled_stage_kinds, "InPlaceRope") == 1
            assert _stage_count(runtime_stage_kinds, "FusedKernel") == 0
            assert _stage_count(compiled_stage_kinds, "FusedKernel") == 0
        else:
            assert _stage_count(runtime_stage_kinds, "FusedKernel") == 1
            assert _stage_count(compiled_stage_kinds, "FusedKernel") == 1
            assert _stage_count(runtime_stage_kinds, "InPlaceRope") == 0
            assert _stage_count(compiled_stage_kinds, "InPlaceRope") == 0


@pytest.mark.cuda
@pytest.mark.performance
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_name)
def test_attention_layer_real_size_throughput(case: AttentionPerfCase, dtype: thor.DataType, record_property):
    if not cudnn_frontend_attention_available():
        pytest.skip("Thor was not built with cuDNN Frontend attention support")

    stream = Stream(Placement(DeviceType.gpu, GPU_NUM))
    program, input_shapes, output_shape, flops_per_launch, metadata = _unpack_built_case(case.builder(dtype))
    compiled_stage_kinds, runtime_stage_kinds = _debug_program_stage_kinds(program, input_shapes, dtype, stream)
    launches, input_bytes, pool_slots = _make_stamped_launch_pool(program, input_shapes, dtype, stream)

    elapsed_s = _benchmark_rotating_launches(launches, stream)
    ms_per_launch = (elapsed_s / MEASURE_ITERS) * 1_000.0
    launches_per_s = MEASURE_ITERS / elapsed_s
    tflops = (flops_per_launch * MEASURE_ITERS) / elapsed_s / 1.0e12
    output_elems_per_s = (prod(output_shape) * MEASURE_ITERS) / elapsed_s
    input_gib_per_s = (input_bytes * MEASURE_ITERS) / elapsed_s / float(1024**3)
    rotating_pool_gib = (input_bytes * pool_slots) / float(1024**3)

    record_property("case", case.name)
    record_property("description", case.description)
    record_property("dtype", _dtype_name(dtype))
    record_property("measure_iters", MEASURE_ITERS)
    record_property("warmup_iters", max(WARMUP_ITERS, pool_slots))
    record_property("initialize_inputs", INITIALIZE_INPUTS)
    record_property("rotating_pool_slots", pool_slots)
    record_property("rotating_input_pool_gib", rotating_pool_gib)
    record_property("input_gib_per_launch", input_bytes / float(1024**3))
    record_property("ms_per_launch", ms_per_launch)
    record_property("launches_per_second", launches_per_s)
    record_property("estimated_tflops", tflops)
    record_property("input_gib_per_second", input_gib_per_s)
    record_property("output_elements_per_second", output_elems_per_s)
    record_property("compiled_stage_kinds", str(compiled_stage_kinds))
    record_property("runtime_stage_kinds", str(runtime_stage_kinds))
    record_property("compiled_fused_kernel_stage_count", _stage_count(compiled_stage_kinds, "FusedKernel"))
    record_property("runtime_fused_kernel_stage_count", _stage_count(runtime_stage_kinds, "FusedKernel"))
    record_property("compiled_attention_stage_count", _stage_count(compiled_stage_kinds, "Attention"))
    record_property("runtime_attention_stage_count", _stage_count(runtime_stage_kinds, "Attention"))
    for key, value in metadata.items():
        record_property(key, value)

    metadata_suffix = ""
    if metadata:
        metadata_suffix = " | " + " | ".join(f"{key}={value}" for key, value in sorted(metadata.items()))

    print(
        f"{case.name} [{_dtype_name(dtype)}]: "
        f"{ms_per_launch:.3f} ms/launch | "
        f"{launches_per_s:,.2f} launches/s | "
        f"{tflops:.3f} estimated TFLOP/s | "
        f"{input_gib_per_s:.3f} input GiB/s | "
        f"rotating pool={pool_slots} slots/{rotating_pool_gib:.2f} GiB"
        f"{metadata_suffix} | "
        f"compiled_stages={_stage_summary(compiled_stage_kinds)} | "
        f"runtime_stages={_stage_summary(runtime_stage_kinds)}")

    if metadata.get("use_rope"):
        assert _stage_count(runtime_stage_kinds, "Attention") == 1
    assert elapsed_s > 0.0
    assert pool_slots >= 2 or EXPLICIT_POOL_SLOTS == "1"
