# Thor einsum primitive-class cost calibration

This opt-in benchmark is the final calibration checkpoint before the six-operand einsum bridge (Patch 8). Its purpose is deliberately narrower than the cuBLASLt shape investigation in `EinsumExactPlannerBenchmark.cpp`.

The planner should model **intrinsic lowering work**, while each backend should make its primitive efficient on the current GPU. Consequently this benchmark calibrates broad operation classes and does **not** teach einsum about selected cuBLASLt algorithms, `wavesCount`, or GPU-specific skinny-GEMM efficiency.

It measures four production paths in FP32:

| primitive | timed production path | planner work unit |
| --- | --- | --- |
| GEMM | `Expression::matmul` -> cuBLASLt | one FMA |
| fused pair product | one `FusedKernel` for `lhs * rhs` | one output/iteration element |
| reduction | central `CubReduction(Sum)` over contiguous width 256 | one reduction input element |
| materialization | `materializeTensorViewAsync` dense D2D path | one copied element (one read + one write) |

The GEMM sweep fixes `m=n=2048` and varies `k`. This makes arithmetic scale approximately linearly while avoiding the small/skinny shape anomalies that motivated the previous diagnostic benchmark. The elementwise, reduction, and materialization sweeps vary total element count over the same broad powers-of-four range.

## Output

`primitive_samples` reports the raw timings and logical throughput.

`primitive_fit_summary` fits the largest points to:

```text
time_ms = intercept_ms + slope_ms_per_work_unit * work_units
```

and reports:

- the fitted fixed-cost intercept;
- the per-work-unit slope;
- R², so a poor linear model is visible rather than hidden;
- a median per-unit cost over the same large points;
- the slope relative to one GEMM FMA;
- the fitted intercept expressed as equivalent GEMM FMAs.

`planner_weight_guidance` normalizes the slopes to GEMM FMA = 1 and prints approximate integer ratios. These values are **calibration evidence only**; the benchmark does not modify the production planner weights. Materialization measures a read+write copy, so the result-write suggestion uses half of the materialization slope as a first-order write-only traffic estimate.

Fixed primitive costs are represented structurally by primitive/group counts (GEMM groups, reduction launches, fused launches, materializations), rather than by cuBLASLt shape-specific timing.

## Build and run

```bash
cmake --build build-release --target thor_einsum_primitive_cost_benchmark -j
./build-release/thor_einsum_primitive_cost_benchmark
```

A three-point smoke run is available when checking a new build:

```bash
./build-release/thor_einsum_primitive_cost_benchmark --quick
```

Options:

```text
--device=N
--quick
--fit-points=N
```

Run the full sweep in Release for planner calibration. Debug timings are useful only for correctness/smoke checks.

The target is `EXCLUDE_FROM_ALL`; normal Thor builds and tests are unaffected.

## Calibrated production policy

The Release calibration on the current Thor reference GPU produced approximate slope ratios of GEMM FMA `1`, fused pair-product element `210`, reduction input element `60`, materialized element `152`, and write-only element `76`.

Production planning intentionally does **not** copy those device-specific values verbatim. `result_write_elements` is already charged separately for every pair result, so the fused pair-product measurement must first remove its output-write component. The resulting broad architecture-class policy is rounded to powers of two:

```text
GEMM FMA                  1
fused element           128   # input traffic + elementwise compute; output write separate
reduction input element  64
materialized element    128   # one read + one write
result write element     64
```

These values preserve the measured ordering and rough ratios while avoiding dependence on one GPU's exact bandwidth/compute balance. Primitive launch overhead is not scalarized from the linear-fit intercepts because the measured intercepts were negative across all four fitted classes, showing that one affine model does not span both launch-bound and throughput-bound regimes reliably. Instead the planner records GEMM groups, fused-kernel operations, reduction operations, and materialization operations explicitly and uses those counts for dominance/tie-breaking when weighted work is otherwise comparable.
