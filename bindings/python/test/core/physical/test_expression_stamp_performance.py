from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod

import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream

GPU_NUM = int(os.getenv("THOR_EXPR_PERF_GPU", "0"))

# Keep these env-tunable so you can easily scale on different machines.
WARMUP_ITERS = int(os.getenv("THOR_EXPR_PERF_WARMUP_ITERS", "25"))
MEASURE_ITERS = int(os.getenv("THOR_EXPR_PERF_MEASURE_ITERS", "200"))

# Choose shapes so flat and broadcast cases produce the same output numel.
FLAT_SHAPE = (33_554_432,)
FLAT_SHAPE_BIG = (2 * 4096, 2 * 2048)  # 33,554,432 elems
FLAT_SHAPE_SMALL = (2 * 2048, 2 * 1024)  # 8,388,608 elems

BCAST_X_SHAPE = (256, 1, 512)
BCAST_Y_SHAPE = (1, 256, 512)
BCAST_Z_SHAPE = (256, 256, 1)
BCAST_OUT_SHAPE = (256, 256, 512)
BCAST_NARROW_SHAPE = BCAST_X_SHAPE
BCAST_WIDE_SHAPE = BCAST_OUT_SHAPE

DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]


def _benchmark_cached_launches(launch_fn, stream: Stream) -> float:
    """
    Returns wall time in seconds for MEASURE_ITERS cached launches only.

    Timing protocol:
      1. One untimed launch to trigger stamp-time specialization / cache fill
      2. One more untimed cached launch to flush any one-time runtime setup
      3. Several untimed warmup launches
      4. Timed repeated cached launches
    """
    # Trigger stamp-specialization / cache fill.
    launch_fn()
    stream.synchronize()

    # First cached launch, still untimed.
    launch_fn()
    stream.synchronize()

    # Warm cache / warm device.
    for _ in range(WARMUP_ITERS):
        launch_fn()
    stream.synchronize()

    start = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
    for _ in range(MEASURE_ITERS):
        launch_fn()
    end = stream.put_event(enable_timing=True, expecting_host_to_wait=True)
    ms: float = end.synchronize_and_report_elapsed_time_ms(start)

    return ms / 1000.0


def _make_stamped_single_launch(program, inputs: dict[str, PhysicalTensor], stream: Stream):
    """
    Stamp once, then return a zero-arg launch function that runs only the cached
    stamped execution plan.
    """
    stamped = program.stamp(inputs, stream)

    # Materialize the output once so the plan is fully wired before timing.
    _ = stamped.output()

    def launch() -> None:
        stamped.run()

    return launch


def _make_stamped_multi_launch(program, inputs: dict[str, PhysicalTensor], stream: Stream):
    """
    Stamp once, then return a zero-arg launch function that runs only the cached
    stamped execution plan.
    """
    stamped = program.stamp(inputs, stream)

    # Materialize outputs once so the plan is fully wired before timing.
    for name in program.output_names():
        _ = stamped.output(name)

    def launch() -> None:
        stamped.run()

    return launch


@dataclass(frozen=True)
class PerfCase:
    name: str
    kind: str  # "single" or "multi"
    builder: callable


def _build_flat_single(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    expr = (x * y) + z

    input_shapes = {
        "x": FLAT_SHAPE,
        "y": FLAT_SHAPE,
        "z": FLAT_SHAPE,
    }
    output_shape = FLAT_SHAPE
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_flat_single_compute_bound(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    num_accums = 24
    dependency_chain_length = 24

    # Build many independent accumulators so the GPU has much more ILP than a
    # single long dependency chain. The final output folds them back together.
    accs = []
    for j in range(num_accums):
        ax = 1.0009765625 + 0.0001220703125 * (j % 4)
        ay = 0.99951171875 - 0.0001220703125 * (j % 4)
        az = 1.001953125 + 0.000244140625 * (j % 4)
        acc = (x * ax) + (y * ay) + (z * az)
        accs.append(acc)

    for i in range(dependency_chain_length):
        a = 1.0001 + 0.0001 * (i % 7)
        b = 0.9999 - 0.0001 * (i % 5)
        c = 0.5 + 0.03125 * (i % 3)
        d = 0.25 + 0.015625 * (i % 4)

        for j in range(num_accums):
            # Slightly vary constants per accumulator so they do not collapse
            # into identical chains.
            aj = a + 0.00001 * (j % 8)
            bj = b - 0.00001 * (j % 8)
            cj = c + 0.0078125 * (j % 4)
            dj = d + 0.00390625 * (j % 4)

            t = accs[j]
            t = (t * aj) + (y * bj)
            t = (t * cj) + (z * dj)
            accs[j] = t

    expr = accs[0]
    for j in range(1, num_accums):
        expr = expr + accs[j]

    input_shapes = {
        "x": FLAT_SHAPE,
        "y": FLAT_SHAPE,
        "z": FLAT_SHAPE,
    }
    output_shape = FLAT_SHAPE
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_broadcast_single(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    expr = (x * y) + z

    input_shapes = {
        "x": BCAST_X_SHAPE,
        "y": BCAST_Y_SHAPE,
        "z": BCAST_Z_SHAPE,
    }
    output_shape = BCAST_OUT_SHAPE
    return ex.compile(expr, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shape


def _build_flat_multi(dtype: thor.DataType):
    a = ex.input("a")
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    xy = x + y
    outs = ex.outputs({
        "sum": xy + a,
        "mix": xy * z * a,
    })

    input_shapes = {
        "a": FLAT_SHAPE,
        "x": FLAT_SHAPE,
        "y": FLAT_SHAPE,
        "z": FLAT_SHAPE,
    }
    output_shapes = {
        "sum": FLAT_SHAPE,
        "mix": FLAT_SHAPE,
    }
    return ex.compile(outs, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shapes


def _build_flat_multi_disjoint_inputs_different_output_shapes(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    d = ex.input("d")

    # Two disjoint flat branches with different output shapes/domains.
    outs = ex.outputs(
        {
            "ab_out": (a * b) + a,  # shape = FLAT_SHAPE_BIG
            "cd_out": (c + d) * d,  # shape = FLAT_SHAPE_SMALL
        })

    input_shapes = {
        "a": FLAT_SHAPE_BIG,
        "b": FLAT_SHAPE_BIG,
        "c": FLAT_SHAPE_SMALL,
        "d": FLAT_SHAPE_SMALL,
    }
    output_shapes = {
        "ab_out": FLAT_SHAPE_BIG,
        "cd_out": FLAT_SHAPE_SMALL,
    }
    return ex.compile(outs, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shapes


def _build_broadcast_multi(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    xy = x + y
    xz = x + z
    outs = ex.outputs({
        "xy": xy,
        "xz": xz,
        "mix": xy * xz,
    })

    input_shapes = {
        "x": BCAST_X_SHAPE,
        "y": BCAST_Y_SHAPE,
        "z": BCAST_Z_SHAPE,
    }
    output_shapes = {
        "xy": BCAST_OUT_SHAPE,
        "xz": BCAST_OUT_SHAPE,
        "mix": BCAST_OUT_SHAPE,
    }
    return ex.compile(outs, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shapes


def _build_broadcast_multi_multiple_shapes_shared_inputs(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    # Two broadcast output domains:
    #   wide   -> [256, 256, 512]
    #   narrow -> [256,   1, 512]
    xy = x + y

    outs = ex.outputs(
        {
            "wide_sum": xy,  # [256, 256, 512]
            "wide_mix": xy * y,  # [256, 256, 512]
            "narrow": x * z,  # [256,   1, 512]
        })

    input_shapes = {
        "x": BCAST_X_SHAPE,  # [256, 1, 512]
        "y": BCAST_Y_SHAPE,  # [1, 256, 512]
        "z": BCAST_X_SHAPE,  # [256, 1, 512]
    }
    output_shapes = {
        "wide_sum": BCAST_WIDE_SHAPE,  # [256, 256, 512]
        "wide_mix": BCAST_WIDE_SHAPE,  # [256, 256, 512]
        "narrow": BCAST_NARROW_SHAPE,  # [256, 1, 512]
    }
    return ex.compile(outs, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shapes


def _build_broadcast_multi_multiple_shapes_disjoint_inputs(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    d = ex.input("d")

    # Two broadcast output domains with disjoint input sets:
    #   wide   from {a, b} -> [256, 256, 512]
    #   narrow from {c, d} -> [256,   1, 512]
    ab = a + b
    cd = c * d

    outs = ex.outputs(
        {
            "wide_sum": ab,  # [256, 256, 512]
            "wide_mix": ab * b,  # [256, 256, 512]
            "narrow_prod": cd,  # [256, 1, 512]
            "narrow_sum": c + d,  # [256, 1, 512]
        })

    input_shapes = {
        "a": BCAST_X_SHAPE,  # [256, 1, 512]
        "b": BCAST_Y_SHAPE,  # [1, 256, 512]
        "c": BCAST_X_SHAPE,  # [256, 1, 512]
        "d": BCAST_X_SHAPE,  # [256, 1, 512]
    }
    output_shapes = {
        "wide_sum": BCAST_WIDE_SHAPE,  # [256, 256, 512]
        "wide_mix": BCAST_WIDE_SHAPE,  # [256, 256, 512]
        "narrow_prod": BCAST_NARROW_SHAPE,  # [256, 1, 512]
        "narrow_sum": BCAST_NARROW_SHAPE,  # [256, 1, 512]
    }
    return ex.compile(outs, device_num=GPU_NUM, use_fast_math=False), input_shapes, output_shapes


CASES = [
    # PerfCase("flat_single_output_compute_bound", "single", _build_flat_single_compute_bound),
    PerfCase("flat_single_output", "single", _build_flat_single),
    PerfCase("flat_multi_output", "multi", _build_flat_multi),
    PerfCase(
        "flat_multi_output_disjoint_inputs_different_output_shapes",
        "multi",
        _build_flat_multi_disjoint_inputs_different_output_shapes,
    ),
    PerfCase("broadcast_single_output", "single", _build_broadcast_single),
    PerfCase("broadcast_multi_output", "multi", _build_broadcast_multi),
    PerfCase(
        "broadcast_multi_output_multiple_shapes_shared_inputs",
        "multi",
        _build_broadcast_multi_multiple_shapes_shared_inputs,
    ),
    PerfCase(
        "broadcast_multi_output_multiple_shapes_disjoint_inputs",
        "multi",
        _build_broadcast_multi_multiple_shapes_disjoint_inputs,
    ),
]


@pytest.mark.cuda
@pytest.mark.performance
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "dtype",
    DTYPES,
    ids=lambda d: str(d).split(".")[-1],
)
def test_expression_kernel_throughput(case: PerfCase, dtype: thor.DataType, record_property):
    """
    Measures repeated stamped execution throughput.

    Important behavior:
      - Compile happens once in the builder.
      - Stamp happens once outside the timed region.
      - Inputs are allocated once outside timing.
      - No host copies or per-call binding occur inside the timed region.
      - Broadcast specialization is fixed after the untimed stamp.
    """
    gpu_placement = Placement(DeviceType.gpu, GPU_NUM)
    stream = Stream(gpu_placement)

    program, input_shapes, output_spec = case.builder(dtype)
    inputs = {
        name: PhysicalTensor(gpu_placement, PhysicalTensor.Descriptor(dtype, shape))
        for name, shape in input_shapes.items()
    }

    if case.kind == "single":
        launch = _make_stamped_single_launch(program, inputs, stream)
        total_output_elems = prod(output_spec)
    else:
        launch = _make_stamped_multi_launch(program, inputs, stream)
        total_output_elems = sum(prod(shape) for shape in output_spec.values())

    elapsed_s = _benchmark_cached_launches(launch, stream)

    launches_per_s = MEASURE_ITERS / elapsed_s
    elems_per_s = (total_output_elems * MEASURE_ITERS) / elapsed_s
    ms_per_launch = (elapsed_s / MEASURE_ITERS) * 1_000.0

    record_property("case", case.name)
    record_property("dtype", str(dtype))
    record_property("measure_iters", MEASURE_ITERS)
    record_property("warmup_iters", WARMUP_ITERS)
    record_property("ms_per_launch", ms_per_launch)
    record_property("launches_per_second", launches_per_s)
    record_property("output_elements_per_second", elems_per_s)

    print(
        f"{case.name}: "
        f"{ms_per_launch:.3f} ms/launch | "
        f"{launches_per_s:,.2f} launches/s | "
        f"{elems_per_s / 1e9:.3f} Goutput-elem/s")

    assert elapsed_s > 0.0
