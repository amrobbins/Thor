# Thor multi-operand einsum execution benchmark

This opt-in benchmark is the execution-performance checkpoint after the multi-operand planner scalability work.
It exercises the **selected production contraction tree through the real Thor `Expression` execution path**; it does not
introduce a benchmark-only einsum executor.

The default `large` profile covers:

- 3-, 4-, and 5-operand exact subset-DP matrix chains;
- the six-operand bridge;
- 7- and 10-operand beam-selected matrix chains;
- a seven-operand batched beam chain;
- a seven-operand branching/cyclic beam network that must use at least one Expression helper lane;
- a seven-operand local-reduction + GEMM mixture that must contain a centralized reduction stage.

Every benchmark case first runs the same equation on all-dimension-2 tensors and compares optimized execution against
`stampGenericReference(...)`. This keeps differential correctness coverage safe even when the performance shape would make
the whole-equation generic broadcast enormous.

For pure matrix and batched-matrix chains, the performance shape is also compared and timed against an intentionally poor
left-to-right matmul tree. This provides a concrete execution-quality baseline without teaching the production planner about
benchmark-specific GPU timings.

The default `large` profile uses `outer=2048` and `inner=512`. This is deliberate: the prior `medium` profile
(`outer=1024`, `inner=64`) could make the preferred contraction tree contain a `64 x 64 x 1024` GEMM, a shape already
known from primitive calibration to be dominated by low GPU utilization rather than mathematical work. The large profile
keeps every matmul in the pure chain fixtures above a 64 Mi-FMA floor and the benchmark reports the minimum selected and
bad-order matmul FMA counts so this property remains observable. The branching fixture is a two-path matrix network
(`ab,bc,cd,de,ef,fg,bg->ae`) using `inner x inner` operands; under the large profile its helper-stream work therefore also
stays above the same 64 Mi-FMA floor instead of reintroducing a tiny connector GEMM. Only the local-reduction dimension
stays deliberately modest for the mixed reduction fixture.

The whole-equation generic performance path is timed only when its estimated broadcast tensor is below
`--max-generic-mib` (256 MiB by default).

## Build and run

```bash
cmake --build build-release --target thor_einsum_multi_operand_execution_benchmark -j
./build-release/thor_einsum_multi_operand_execution_benchmark
```

Useful focused runs:

```bash
./build-release/thor_einsum_multi_operand_execution_benchmark --case=beam10
./build-release/thor_einsum_multi_operand_execution_benchmark --case=branching --verbose-plan
./build-release/thor_einsum_multi_operand_execution_benchmark --case=reduction_gemm --verbose-plan
```

Use the medium, small, or tiny profiles for faster iteration and to make more whole-equation generic cases safe enough to time:

```bash
./build-release/thor_einsum_multi_operand_execution_benchmark --profile=medium
./build-release/thor_einsum_multi_operand_execution_benchmark --profile=small
./build-release/thor_einsum_multi_operand_execution_benchmark --profile=tiny --max-generic-mib=512
```

Timing controls:

```text
--samples=N
--iterations=N
--warmup=N
```

Other options:

```text
--device=N
--profile=tiny|small|medium|large
--case=SUBSTRING
--max-generic-mib=N
--verbose-plan
```

CSV output reports selected latency, bad-left-to-right latency/speedup where applicable, generic latency/speedup when safely
measured, the selected planner cost, stage counts, helper-lane matmul diagnostics, and minimum matmul FMA counts. Under the
large profile the branching fixture fails if its selected plan contains a matmul below the 64 Mi-FMA performance floor or if
it no longer uses an Expression helper lane. The reduction/GEMM fixture fails if it no longer contains a central reduction
stage.

The target is `EXCLUDE_FROM_ALL`, so it adds no work to ordinary Thor builds or tests.
