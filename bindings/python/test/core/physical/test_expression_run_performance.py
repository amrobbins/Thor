from __future__ import annotations

import os
import time
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
FLAT_SHAPE_BIG = (2 * 4096, 2 * 2048)
FLAT_SHAPE_SMALL = (2 * 2048, 2 * 1024)

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


def _execute_single(expr, inputs: dict[str, PhysicalTensor], output: PhysicalTensor, stream: Stream) -> None:
    """
    Replace with your exact single-output execution API if needed.

    Common shapes this may take in your bindings:
      expr.run(inputs, output, stream)
      expr.run(inputs=inputs, output=output, stream=stream)
    """
    try:
        expr.run(inputs, output, stream)
        return
    except TypeError:
        pass

    try:
        expr.run(inputs=inputs, output=output, stream=stream)
        return
    except TypeError as exc:
        raise RuntimeError("Wire _execute_single() to your exact Python binding call signature.") from exc


def _execute_multi(outs, inputs: dict[str, PhysicalTensor], outputs: dict[str, PhysicalTensor], stream: Stream) -> None:
    """
    Replace with your exact multi-output execution API if needed.

    Common shapes this may take in your bindings:
      outs.run(inputs, outputs, stream)
      outs.run(inputs=inputs, outputs=outputs, stream=stream)
    """
    try:
        outs.run(inputs, outputs, stream)
        return
    except TypeError:
        pass

    try:
        outs.run(inputs=inputs, outputs=outputs, stream=stream)
        return
    except TypeError as exc:
        raise RuntimeError("Wire _execute_multi() to your exact Python binding call signature.") from exc


def _benchmark_cached_launches(launch_fn, stream: Stream) -> float:
    """
    Returns wall time in seconds for MEASURE_ITERS cached launches only.

    Timing protocol:
      1. One untimed launch to trigger JIT/cache population
      2. One more untimed cached launch to flush any one-time runtime setup
      3. Several untimed warmup launches
      4. Timed repeated cached launches
    """
    # Trigger JIT compilation / cache fill.
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


@dataclass(frozen=True)
class PerfCase:
    name: str
    kind: str  # "single" or "multi"
    builder: callable


def _build_flat_single():
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
    return ex.compile(expr), input_shapes, output_shape


def _build_flat_single_compute_bound():
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    num_accums = 16
    dependency_chain_length = 4

    # Build 16 independent accumulators so the GPU has much more ILP than a
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
    return ex.compile(expr), input_shapes, output_shape


def _build_broadcast_single():
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
    return ex.compile(expr), input_shapes, output_shape


def _build_flat_multi():
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
    return outs.compile(), input_shapes, output_shapes


def _build_flat_multi_disjoint_inputs_different_output_shapes():
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
        "a": FLAT_SHAPE,
        "b": FLAT_SHAPE,
        "c": FLAT_SHAPE,
        "d": FLAT_SHAPE,
    }
    output_shapes = {
        "ab_out": FLAT_SHAPE,
        "cd_out": FLAT_SHAPE,
    }
    return outs.compile(), input_shapes, output_shapes


def _build_broadcast_multi():
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
    return outs.compile(), input_shapes, output_shapes


def _build_broadcast_multi_multiple_shapes_shared_inputs():
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    # Two broadcast output domains:
    #   wide   -> [128, 256, 256]
    #   narrow -> [128,   1, 256]
    xy = x + y

    outs = ex.outputs(
        {
            "wide_sum": xy,  # [128, 256, 256]
            "wide_mix": xy * y,  # [128, 256, 256]
            "narrow": x * z,  # [128,   1, 256]
        })

    input_shapes = {
        "x": BCAST_X_SHAPE,  # [128, 1, 256]
        "y": BCAST_Y_SHAPE,  # [1, 256, 256]
        "z": BCAST_X_SHAPE,  # [128, 1, 256]
    }
    output_shapes = {
        "wide_sum": BCAST_WIDE_SHAPE,  # [128, 256, 256]
        "wide_mix": BCAST_WIDE_SHAPE,  # [128, 256, 256]
        "narrow": BCAST_NARROW_SHAPE,  # [128, 1, 256]
    }
    return outs.compile(), input_shapes, output_shapes


def _build_broadcast_multi_multiple_shapes_disjoint_inputs():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    d = ex.input("d")

    # Two broadcast output domains with disjoint input sets:
    #   wide   from {a, b} -> [128, 256, 256]
    #   narrow from {c, d} -> [128,   1, 256]
    ab = a + b
    cd = c * d

    outs = ex.outputs(
        {
            "wide_sum": ab,  # [128, 256, 256]
            "wide_mix": ab * b,  # [128, 256, 256]
            "narrow_prod": cd,  # [128, 1, 256]
            "narrow_sum": c + d,  # [128, 1, 256]
        })

    input_shapes = {
        "a": BCAST_X_SHAPE,  # [128, 1, 256]
        "b": BCAST_Y_SHAPE,  # [1, 256, 256]
        "c": BCAST_X_SHAPE,  # [128, 1, 256]
        "d": BCAST_X_SHAPE,  # [128, 1, 256]
    }
    output_shapes = {
        "wide_sum": BCAST_WIDE_SHAPE,  # [128, 256, 256]
        "wide_mix": BCAST_WIDE_SHAPE,  # [128, 256, 256]
        "narrow_prod": BCAST_NARROW_SHAPE,  # [128, 1, 256]
        "narrow_sum": BCAST_NARROW_SHAPE,  # [128, 1, 256]
    }
    return outs.compile(), input_shapes, output_shapes


CASES = [
    PerfCase("flat_single_output_compute_bound", "single", _build_flat_single_compute_bound),
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
    Measures cached kernel throughput only.

    Important behavior:
      - Flat kernels are shape-agnostic after initial compile/cache fill.
      - Broadcast kernels are shape-specialized, so the benchmark keeps shapes fixed
        after the untimed compile-triggering launch.
      - Inputs/outputs are allocated once outside timing so allocation is not measured.
      - No host copies occur inside the timed region.
    """
    stream = Stream(Placement(DeviceType.gpu, GPU_NUM))

    program, input_shapes, output_spec = case.builder()
    inputs = {
        name: PhysicalTensor(Placement(DeviceType.gpu, GPU_NUM), PhysicalTensor.Descriptor(dtype, shape))
        for name, shape in input_shapes.items()
    }

    if case.kind == "single":
        output = PhysicalTensor(Placement(DeviceType.gpu, GPU_NUM), PhysicalTensor.Descriptor(dtype, output_spec))
        total_output_elems = prod(output_spec)

        def launch():
            _execute_single(program, inputs, output, stream)

    else:
        outputs = {
            name: PhysicalTensor(Placement(DeviceType.gpu, GPU_NUM), PhysicalTensor.Descriptor(dtype, shape))
            for name, shape in output_spec.items()
        }
        total_output_elems = sum(prod(shape) for shape in output_spec.values())

        def launch():
            _execute_multi(program, inputs, outputs, stream)

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
