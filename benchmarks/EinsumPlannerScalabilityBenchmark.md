# Thor einsum planner scalability benchmark

This opt-in benchmark measures the **CPU planning cost and beam-search state volume** of Thor's production multi-operand einsum planner. It does not allocate tensors, launch CUDA work, or measure execution latency. Its purpose is to validate that the bounded beam planner remains tractable as operand count grows, make the effect of the production width-32 cutoff directly observable, and support controlled diagnostic comparisons against alternate widths without changing production policy.

The default sweep uses matrix chains with:

```text
7, 10, 20, 40 operands
```

Each chain uses deterministic, varied dimensions in the range 2-20 so contraction-order decisions are nontrivial without risking cost-model overflow. The benchmark repeats the same planner invocation three times by default, verifies that the complete `describeBeamContraction(...)` output is identical across samples, and reports median/best/worst CPU wall-clock planning time.

## State counters

Each row reports the cumulative beam diagnostics already produced by the production planner plus the width-cutoff counter added for this benchmark:

- `expanded_states`: retained beam states expanded across all heuristic levels;
- `generated_states`: successful physical next-state candidates before physical-signature deduplication;
- `deduplicated_states`: generated candidates merged into an already-seen physical frontier state;
- `unique_states = generated_states - deduplicated_states`;
- `truncated_states`: physically unique states discarded only because the sorted frontier exceeded the row's `beam_width`;
- `retained_states`: physically unique states retained after the width cutoff, accumulated across levels;
- `truncated_unique_percent`: `truncated_states / unique_states`;
- `deferred_disconnected_pairs`: unordered active pairs skipped before physical candidate generation because they would prematurely form an outer-product expansion while both sides still have connected work. Persistent passthrough labels such as batch labels do not by themselves make a pair contraction useful;
- `exact_tails`: retained five-active-operand frontiers completed by the exact planner.

For every completed beam plan the benchmark verifies:

```text
unique_states == truncated_states + retained_states
exact_tails <= beam_width
```

`truncated_states` is intentionally separate from deduplication. A high deduplication count means physical-state canonicalization is removing redundant work. A high truncation count means many **distinct** candidate states survive canonicalization but are being rejected by the beam-width policy. That is the useful signal when deciding whether width 32 deserves a quality/performance comparison against another width in a later experiment.

The benchmark also reports the selected plan's modeled execution cost, peak intermediate elements, and primitive operation counts so scalability observations can be compared without losing sight of plan quality.

## Families

`matrix_chain` is the default because it supports long connected contraction graphs while requiring only two labels per operand.

`batched_chain` adds one persistent batch label to every operand and exercises the batched-GEMM physical planning path without changing the logical chain topology. Nonadjacent operands therefore share the batch label but no reducible contraction label; the planner should recognize those as premature batched outer products rather than treating the shared passthrough label as useful connectivity.

Use `--family=all` to run both. Each CSV row now includes `beam_width`. The ordinary planner continues to use `EinsumPlanner::DEFAULT_BEAM_WIDTH == 32`; alternate widths are available only through the explicitly diagnostics-only planner entry point used by this benchmark.

## Build and run

```bash
cmake --build build-release --target thor_einsum_planner_scalability_benchmark -j
./build-release/thor_einsum_planner_scalability_benchmark
```

Focused runs:

```bash
./build-release/thor_einsum_planner_scalability_benchmark --operands=40 --samples=5
./build-release/thor_einsum_planner_scalability_benchmark --family=batched_chain
./build-release/thor_einsum_planner_scalability_benchmark --family=all --operands=7,10,20,40
./build-release/thor_einsum_planner_scalability_benchmark --operands=20 --verbose-plan
./build-release/thor_einsum_planner_scalability_benchmark --operands=10,20 --samples=1 --beam-widths=16,32,64,128
./build-release/thor_einsum_planner_scalability_benchmark --operands=40 --samples=1 --beam-widths=16,32,64
```

For width-quality experiments, start with one timing sample. Wider beams intentionally expand more frontier states, so a full `40 x 128` sweep can be substantially slower. Compare `estimated_execution_units`, `peak_intermediate_elements`, and primitive counts across widths; if they remain identical, the extra planning time bought no modeled execution-quality improvement for that case.

Options:

```text
--family=matrix_chain|batched_chain|all
--operands=7,10,20,40
--samples=N
--beam-widths=16,32,64,128
--verbose-plan
```

The equation generator uses Thor's ASCII-letter einsum syntax. Matrix-chain cases therefore support at most 51 operands, and batched-chain cases at most 50 because the batched family reserves one label for the persistent batch axis. This is only a benchmark-generator limit; Thor's public einsum operand limit remains 63.

The target is `EXCLUDE_FROM_ALL`, so normal Thor builds and tests do not pay for it.
