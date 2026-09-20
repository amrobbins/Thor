# Thor profiling cases

This directory contains repeatable, representative workloads intended for visual
profiling with tools such as Nsight Systems.  Profiling cases are deliberately
separate from correctness tests and microbenchmarks:

- tests answer whether behavior is correct;
- benchmarks answer how fast one operation is;
- profiling cases make the end-to-end execution timeline easy to inspect and
  compare after runtime changes.

Each case should keep setup and data preparation outside the captured interval,
use deterministic settings where practical, write large generated reports under
`build/profiling/`, and document the timeline invariants it is intended to
exercise.

The first case is [`alexnet_training`](alexnet_training/README.md), which is the
runtime scheduling reference workload for dense, pipelined training.

## Local profiling data

Reusable datasets and preprocessing caches belong under `profiling/data/`. The
whole directory is intentionally ignored by Git and is created by profiling
cases as needed. Keeping data here makes repeated profiles reproducible without
redownloading while keeping generated datasets out of source control.
