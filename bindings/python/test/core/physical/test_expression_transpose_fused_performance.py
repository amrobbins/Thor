from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod
from typing import Callable

import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream

GPU_NUM = int(os.getenv("THOR_EXPR_PERF_GPU", "0"))

# Keep these env-tunable so you can scale the benchmark on different machines.
# These fall back to the existing expression-performance env vars when present.
WARMUP_ITERS = int(os.getenv(
    "THOR_EXPR_TRANSPOSE_PERF_WARMUP_ITERS",
    os.getenv("THOR_EXPR_PERF_WARMUP_ITERS", "25"),
))
MEASURE_ITERS = int(
    os.getenv(
        "THOR_EXPR_TRANSPOSE_PERF_MEASURE_ITERS",
        os.getenv("THOR_EXPR_PERF_MEASURE_ITERS", "200"),
    ))

# Match the large flat-performance suite's element count for the main aligned
# transpose cases: 8192 * 4096 = 33,554,432 elements.
TRANSPOSE_ALIGNED_SHAPE = (
    int(os.getenv("THOR_EXPR_TRANSPOSE_PERF_ROWS", "8192")),
    int(os.getenv("THOR_EXPR_TRANSPOSE_PERF_COLS", "4096")),
)

# Intentionally not divisible by TILE_DIM / packed-tile widths. This exercises
# scalar edge handling in the same benchmark suite as the fast aligned path.
TRANSPOSE_RAGGED_SHAPE = (
    int(os.getenv("THOR_EXPR_TRANSPOSE_PERF_RAGGED_ROWS", "8193")),
    int(os.getenv("THOR_EXPR_TRANSPOSE_PERF_RAGGED_COLS", "4097")),
)

BROADCAST_ROWS = int(os.getenv("THOR_EXPR_TRANSPOSE_PERF_BCAST_ROWS", "8192"))
BROADCAST_COLS = int(os.getenv("THOR_EXPR_TRANSPOSE_PERF_BCAST_COLS", "4096"))

DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]


def _dtype_name(dtype: thor.DataType) -> str:
    return str(dtype).split(".")[-1]


def _bytes_per_element(dtype: thor.DataType) -> int:
    if dtype == thor.DataType.fp32:
        return 4
    if dtype in (thor.DataType.fp16, thor.DataType.bf16):
        return 2
    if dtype in (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2):
        return 1
    raise AssertionError(f"Unhandled dtype: {dtype}")


def _benchmark_cached_launches(launch_fn: Callable[[], None], stream: Stream) -> float:
    """
    Returns wall time in seconds for MEASURE_ITERS cached stamped launches only.

    Timing protocol:
      1. One untimed launch to trigger stamp-time specialization / cache fill
      2. One more untimed cached launch to flush any one-time runtime setup
      3. Several untimed warmup launches
      4. Timed repeated cached launches
    """
    launch_fn()
    stream.synchronize()

    launch_fn()
    stream.synchronize()

    for _ in range(WARMUP_ITERS):
        launch_fn()
    stream.synchronize()

    start = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
    for _ in range(MEASURE_ITERS):
        launch_fn()
    end = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
    ms: float = end.synchronize_and_report_elapsed_time_ms(start)
    return ms / 1000.0


def _make_stamped_launch(program, inputs: dict[str, PhysicalTensor], stream: Stream):
    stamped = program.stamp(inputs, stream)

    # Materialize the output once so allocation and plan wiring stay outside the
    # timed region. The timed loop below should be cached execution only.
    _ = stamped.output()

    def launch() -> None:
        stamped.run()

    return launch


@dataclass(frozen=True)
class PerfCase:
    name: str
    builder: Callable[[thor.DataType, thor.DataType], tuple[object, dict[str, tuple[int, ...]], tuple[int, ...]]]
    logical_output_shape: Callable[[], tuple[int, ...]]
    input_reads_per_output: int
    expects_fused_transpose: bool


def _arithmetic(x, y):
    return (x * 1.25) + (y * -0.5) + ((x - y) * 0.125) - 0.25


def _build_flat_no_transpose(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = _arithmetic(x, y).with_output_dtype(output_dtype)

    input_shapes = {
        "x": TRANSPOSE_ALIGNED_SHAPE,
        "y": TRANSPOSE_ALIGNED_SHAPE,
    }
    output_shape = TRANSPOSE_ALIGNED_SHAPE
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_flat_transpose_aligned(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = _arithmetic(x, y).with_output_dtype(output_dtype).transpose()

    input_shapes = {
        "x": TRANSPOSE_ALIGNED_SHAPE,
        "y": TRANSPOSE_ALIGNED_SHAPE,
    }
    output_shape = (TRANSPOSE_ALIGNED_SHAPE[1], TRANSPOSE_ALIGNED_SHAPE[0])
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_flat_transpose_ragged(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = _arithmetic(x, y).with_output_dtype(output_dtype).transpose()

    input_shapes = {
        "x": TRANSPOSE_RAGGED_SHAPE,
        "y": TRANSPOSE_RAGGED_SHAPE,
    }
    output_shape = (TRANSPOSE_RAGGED_SHAPE[1], TRANSPOSE_RAGGED_SHAPE[0])
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_broadcast_outer_no_transpose(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = _arithmetic(x, y).with_output_dtype(output_dtype)

    input_shapes = {
        "x": (BROADCAST_ROWS, 1),
        "y": (1, BROADCAST_COLS),
    }
    output_shape = (BROADCAST_ROWS, BROADCAST_COLS)
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_broadcast_outer_transpose(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = _arithmetic(x, y).with_output_dtype(output_dtype).transpose()

    input_shapes = {
        "x": (BROADCAST_ROWS, 1),
        "y": (1, BROADCAST_COLS),
    }
    output_shape = (BROADCAST_COLS, BROADCAST_ROWS)
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_broadcast_row_transpose(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = _arithmetic(x, y).with_output_dtype(output_dtype).transpose()

    input_shapes = {
        "x": (BROADCAST_ROWS, BROADCAST_COLS),
        "y": (1, BROADCAST_COLS),
    }
    output_shape = (BROADCAST_COLS, BROADCAST_ROWS)
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _aligned_output_shape() -> tuple[int, ...]:
    return TRANSPOSE_ALIGNED_SHAPE


def _aligned_transposed_output_shape() -> tuple[int, ...]:
    return (TRANSPOSE_ALIGNED_SHAPE[1], TRANSPOSE_ALIGNED_SHAPE[0])


def _ragged_transposed_output_shape() -> tuple[int, ...]:
    return (TRANSPOSE_RAGGED_SHAPE[1], TRANSPOSE_RAGGED_SHAPE[0])


def _broadcast_output_shape() -> tuple[int, ...]:
    return (BROADCAST_ROWS, BROADCAST_COLS)


def _broadcast_transposed_output_shape() -> tuple[int, ...]:
    return (BROADCAST_COLS, BROADCAST_ROWS)


CASES = [
    PerfCase(
        "flat_fused_arithmetic_no_transpose_aligned",
        _build_flat_no_transpose,
        _aligned_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=False,
    ),
    PerfCase(
        "flat_fused_arithmetic_transpose_aligned",
        _build_flat_transpose_aligned,
        _aligned_transposed_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=True,
    ),
    PerfCase(
        "flat_fused_arithmetic_transpose_ragged",
        _build_flat_transpose_ragged,
        _ragged_transposed_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=True,
    ),
    PerfCase(
        "broadcast_outer_fused_arithmetic_no_transpose",
        _build_broadcast_outer_no_transpose,
        _broadcast_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=False,
    ),
    PerfCase(
        "broadcast_outer_fused_arithmetic_transpose",
        _build_broadcast_outer_transpose,
        _broadcast_transposed_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=True,
    ),
    PerfCase(
        "broadcast_row_fused_arithmetic_transpose",
        _build_broadcast_row_transpose,
        _broadcast_transposed_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=True,
    ),
]


@pytest.mark.cuda
@pytest.mark.performance
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_name)
def test_fused_transpose_kernel_throughput(case: PerfCase, dtype: thor.DataType, record_property):
    """
    Measures cached stamped execution throughput for fused arithmetic kernels that
    materialize either the normal layout or the new tiled-transpose layout.

    This intentionally keeps compile, stamp, allocation, output materialization,
    and host transfers outside the timed region. The timed section is only
    repeated `stamped.run()` calls on the cached plan.

    Useful one-off invocation:
      pytest -m performance bindings/python/test/core/physical/test_expression_transpose_fused_performance.py -s
    """
    input_dtype = dtype
    output_dtype = dtype

    gpu_placement = Placement(DeviceType.gpu, GPU_NUM)
    stream = Stream(gpu_placement)

    program, input_shapes, output_shape = case.builder(input_dtype, output_dtype)
    inputs = {
        name: PhysicalTensor(gpu_placement, PhysicalTensor.Descriptor(input_dtype, list(shape)))
        for name, shape in input_shapes.items()
    }

    stage_kinds = program._debug_stage_kinds(inputs)
    assert stage_kinds == ["FusedKernel"]

    launch = _make_stamped_launch(program, inputs, stream)
    elapsed_s = _benchmark_cached_launches(launch, stream)

    output_elements = prod(output_shape)
    launches_per_s = MEASURE_ITERS / elapsed_s
    output_elems_per_s = (output_elements * MEASURE_ITERS) / elapsed_s
    ms_per_launch = (elapsed_s / MEASURE_ITERS) * 1_000.0

    input_bpe = _bytes_per_element(input_dtype)
    output_bpe = _bytes_per_element(output_dtype)
    effective_bytes_per_launch = output_elements * (case.input_reads_per_output * input_bpe + output_bpe)
    effective_gib_per_s = (effective_bytes_per_launch * MEASURE_ITERS) / elapsed_s / (1024.0**3)

    record_property("case", case.name)
    record_property("dtype", str(dtype))
    record_property("input_dtype", str(input_dtype))
    record_property("output_dtype", str(output_dtype))
    record_property("output_shape", str(tuple(output_shape)))
    record_property("output_elements", output_elements)
    record_property("measure_iters", MEASURE_ITERS)
    record_property("warmup_iters", WARMUP_ITERS)
    record_property("ms_per_launch", ms_per_launch)
    record_property("launches_per_second", launches_per_s)
    record_property("output_elements_per_second", output_elems_per_s)
    record_property("effective_gib_per_second", effective_gib_per_s)
    record_property("stage_kinds", str(stage_kinds))
    record_property("expects_fused_transpose", case.expects_fused_transpose)

    print(
        f"{case.name} [{_dtype_name(dtype)}]: "
        f"{ms_per_launch:.3f} ms/launch | "
        f"{launches_per_s:,.2f} launches/s | "
        f"{output_elems_per_s / 1e9:.3f} Goutput-elem/s | "
        f"{effective_gib_per_s:.3f} effective GiB/s")

    assert elapsed_s > 0.0


MIXED_DTYPE_CASES = [
    pytest.param(thor.DataType.fp16, thor.DataType.bf16, id="fp16_to_bf16"),
    pytest.param(thor.DataType.bf16, thor.DataType.fp16, id="bf16_to_fp16"),
    pytest.param(thor.DataType.fp8_e4m3, thor.DataType.fp16, id="fp8_e4m3_to_fp16"),
    pytest.param(thor.DataType.fp8_e5m2, thor.DataType.fp32, id="fp8_e5m2_to_fp32"),
]

MIXED_DTYPE_PERF_CASES = [
    PerfCase(
        "flat_fused_arithmetic_transpose_aligned",
        _build_flat_transpose_aligned,
        _aligned_transposed_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=True,
    ),
    PerfCase(
        "broadcast_outer_fused_arithmetic_transpose",
        _build_broadcast_outer_transpose,
        _broadcast_transposed_output_shape,
        input_reads_per_output=2,
        expects_fused_transpose=True,
    ),
]


@pytest.mark.cuda
@pytest.mark.performance
@pytest.mark.parametrize("case", MIXED_DTYPE_PERF_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("input_dtype,output_dtype", MIXED_DTYPE_CASES)
def test_fused_transpose_mixed_dtype_kernel_throughput(
    case: PerfCase,
    input_dtype: thor.DataType,
    output_dtype: thor.DataType,
    record_property,
):
    """Representative conversion benchmarks for the fused transpose paths."""
    gpu_placement = Placement(DeviceType.gpu, GPU_NUM)
    stream = Stream(gpu_placement)

    program, input_shapes, output_shape = case.builder(input_dtype, output_dtype)
    inputs = {
        name: PhysicalTensor(gpu_placement, PhysicalTensor.Descriptor(input_dtype, list(shape)))
        for name, shape in input_shapes.items()
    }

    stage_kinds = program._debug_stage_kinds(inputs)
    assert stage_kinds == ["FusedKernel"]

    launch = _make_stamped_launch(program, inputs, stream)
    elapsed_s = _benchmark_cached_launches(launch, stream)

    output_elements = prod(output_shape)
    launches_per_s = MEASURE_ITERS / elapsed_s
    output_elems_per_s = (output_elements * MEASURE_ITERS) / elapsed_s
    ms_per_launch = (elapsed_s / MEASURE_ITERS) * 1_000.0

    input_bpe = _bytes_per_element(input_dtype)
    output_bpe = _bytes_per_element(output_dtype)
    effective_bytes_per_launch = output_elements * (case.input_reads_per_output * input_bpe + output_bpe)
    effective_gib_per_s = (effective_bytes_per_launch * MEASURE_ITERS) / elapsed_s / (1024.0**3)

    record_property("case", case.name)
    record_property("input_dtype", str(input_dtype))
    record_property("output_dtype", str(output_dtype))
    record_property("output_shape", str(tuple(output_shape)))
    record_property("output_elements", output_elements)
    record_property("measure_iters", MEASURE_ITERS)
    record_property("warmup_iters", WARMUP_ITERS)
    record_property("ms_per_launch", ms_per_launch)
    record_property("launches_per_second", launches_per_s)
    record_property("output_elements_per_second", output_elems_per_s)
    record_property("effective_gib_per_second", effective_gib_per_s)
    record_property("stage_kinds", str(stage_kinds))

    print(
        f"{case.name} [{_dtype_name(input_dtype)}->{_dtype_name(output_dtype)}]: "
        f"{ms_per_launch:.3f} ms/launch | "
        f"{launches_per_s:,.2f} launches/s | "
        f"{output_elems_per_s / 1e9:.3f} Goutput-elem/s | "
        f"{effective_gib_per_s:.3f} effective GiB/s")

    assert elapsed_s > 0.0
