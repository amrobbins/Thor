# Thor exact einsum planner calibration benchmark

This opt-in benchmark is the Patch 7 calibration checkpoint before the six-operand bridge. It now measures two distinct objectives that matter to Thor:

1. **isolated latency** for one model/execution path;
2. **shared-GPU throughput** when two independent copies execute concurrently on separate caller streams.

The second objective matters because a narrow GEMM can leave SM capacity available for another model or another independent execution branch. A contraction tree that is slightly slower in isolation can therefore still be the better system-throughput choice if it consumes materially less of the GPU.

## Chain sweep

For each 3-, 4-, and 5-operand alternating matrix chain the benchmark sweeps five profiles:

| profile | outer dimension | inner dimension |
| --- | ---: | ---: |
| `tiny` | 100 | 2 |
| `small` | 256 | 16 |
| `medium` | 1024 | 64 |
| `large` | 2048 | 128 |
| `xlarge` | 4096 | 256 |

Every case always measures:

1. `selected_exact`: the production exact contraction tree lowered into one Expression DAG;
2. `bad_left_to_right`: a deliberately fixed left-to-right matrix chain compiled as one Expression DAG.

The preserved `whole_equation_generic` broadcast-product + reduction path is measured only when its estimated full broadcast tensor is below a safety cap. The default cap is 256 MiB and can be changed with `--max-generic-mib=N`.

Every stamped Matmul stage now reports the selected cuBLASLt kernel calibration metadata:

- logical `m`, `n`, `k`, and batch count;
- selected kernel `wavesCount`;
- `sm_pressure_proxy = min(1, wavesCount)`;
- measured kernel-picker runtime retained by Thor;
- measured picker TFLOP/s;
- workspace bytes and algorithm ID;
- Expression execution lane/dependency count.

`wavesCount` is treated only as a **resource-pressure signal**, not as a latency score. Values below one wave are especially interesting because they indicate that the selected kernel does not need a full device-wide wave of thread blocks and may leave useful capacity for concurrent work.

For each exact and bad tree the benchmark also runs two independently stamped copies concurrently on two caller streams. It reports:

- `dual_per_plan_ms`: pair makespan divided by two completed plans;
- `dual_throughput_scale = isolated_ms / dual_per_plan_ms`.

A scale near `1` means a second concurrent copy adds little aggregate throughput; a scale approaching `2` means two copies can overlap nearly perfectly. This is the direct throughput measurement to compare with `wavesCount`/SM-pressure proxies.

## Focused GEMM shape sweep

`--gemm-shape-sweep` bypasses the einsum chain cases and benchmarks the individual GEMM shapes responsible for the interesting contraction-order crossover.

The sweep uses:

```text
outer = 256, 512, 1024, 2048, 4096
inner = 16, 32, 48, 64, 80, 96, 128, 192, 256
```

and four representative stage shapes:

| kind | GEMM shape `(m,n,k)` | role |
| --- | --- | --- |
| `bottleneck_contract` | `(inner, inner, outer)` | cheap contraction that collapses a large shared dimension |
| `skinny_expand` | `(outer, inner, inner)` | skinny exact-tree follow-up |
| `wide_expand` | `(outer, outer, inner)` | arithmetic-heavy but GPU-friendly wide GEMM |
| `wide_reduce` | `(outer, inner, outer)` | wide intermediate contracted back down |

Each focused GEMM reports both isolated latency and two-stream throughput scaling, along with the selected kernel's waves, picker runtime, TFLOP/s, workspace, and algorithm ID.

Because the full 180-shape grid can cause substantial one-time cuBLASLt kernel-selection work, start with the suspicious region:

```bash
./build-release/thor_einsum_exact_planner_benchmark --gemm-shape-sweep --outer=1024
```

Then compare larger outer dimensions or a single shape family:

```bash
./build-release/thor_einsum_exact_planner_benchmark --gemm-shape-sweep --outer=2048
./build-release/thor_einsum_exact_planner_benchmark --gemm-shape-sweep --shape-kind=bottleneck_contract
```

## Build and common runs

```bash
cmake --build build-release --target thor_einsum_exact_planner_benchmark -j
./build-release/thor_einsum_exact_planner_benchmark
```

Useful focused chain runs:

```bash
./build-release/thor_einsum_exact_planner_benchmark --size=medium
./build-release/thor_einsum_exact_planner_benchmark --case=five_operand
./build-release/thor_einsum_exact_planner_benchmark --case=five_operand --size=large --verbose-plan
```

Disable the dual-stream throughput measurement when only isolated latency is needed:

```bash
./build-release/thor_einsum_exact_planner_benchmark --no-dual-stream
```

Options:

```text
--device=N
--case=SUBSTRING
--size=SUBSTRING
--max-generic-mib=N
--verbose-plan
--no-dual-stream
--gemm-shape-sweep
--shape-kind=SUBSTRING
--outer=N
--inner=N
```

The target remains `EXCLUDE_FROM_ALL`; it adds no work to normal Thor builds or tests.
