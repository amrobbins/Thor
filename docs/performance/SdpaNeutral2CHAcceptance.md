# SERIALIZATION-OWNERSHIP-2C-H — acceptance gate

**Baseline:** `ThorKernels738.zip` (the user-tested G4-C7 source). This gate adds a test runner and this document only. It does **not** modify production SDPA, serialization, cuDNN plans, kernels, binding-map ownership, or workspace behavior.

## Scope and acceptance boundary

1. **Core:** exactly 39 registered Foundation-A through D1 and 2C-E1 through G3 cases; all must **execute and pass** (zero skip/disabled/zero match). These include saved/reloaded B=2/B=7, forward/backward numerical correctness, dropout, and runtime changing offsets without restamp.
2. **Legacy API:** all existing `AttentionApi.*` and `RaggedAttentionArchitectureGuard.*` cases, including the legacy dense/fixed SDPA and the unchanged composite Attention implementation. Their results are compatibility evidence, **not** 2D neutrality acceptance.
3. **cuDNN integration:** descriptor, ragged metadata, mixed ragged, workspace, concurrency, and frontend plan selection tests.
4. **Graph mechanism:** `CudaGraphTest.DeviceUpdatedGridCanUseRaggedOffsetsLastElementAlias` only. Its success **does not prove whole-network capture**.

A skipped test, a missing case, a failed case, or a zero-match suite is a **failed gate**, not a pass. On failure, retain all logs/XML and report the actual failure to the control hub. Do not change production SDPA or start 2D to work around a failing 2C-H test.

## What the workstation operator must run

From the **patched** `~/Thor` repository root. Use an optimized Release build for performance-related compatibility. Disable G4-C3 allocation instrumentation for normal acceptance runs:

```bash
cmake -S . -B build-sdpa-perf -DTHOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=OFF
cmake --build build-sdpa-perf --target thor_tests thor_sdpa_public_layer_benchmark -j 8

# Cheap discovery: verifies that the 39 tests and the compatibility suites exist.
python3 scripts/sdpa_2ch_acceptance.py \
  --binary build-sdpa-perf/thor_tests --phase all --list-only \
  --output sdpa-2ch-discovery

# Core acceptance first; this step must report 39/39, zero skips.
python3 scripts/sdpa_2ch_acceptance.py \
  --binary build-sdpa-perf/thor_tests --phase core \
  --output sdpa-2ch-core

# Historical and cuDNN compatibility, independently reported.
python3 scripts/sdpa_2ch_acceptance.py \
  --binary build-sdpa-perf/thor_tests --phase compat \
  --output sdpa-2ch-compat
```

The runner prints `GATE_PASS` on a passing run, and writes `summary.md`, `summary.json`, individual XML files, and console logs. It prints `GATE_FAIL` and exits nonzero for zero-match, skipped, failed, or incomplete execution. The two phases together cover all categories. **Do not discard failed logs.**

For a separate normal Debug build, the same runner can use `--binary build/thor_tests` without rebuilding the Release tree. The Release run above is the primary gate.

If you have a `check-cpp` target in your build, you may run it separately; that target is not substituted for these specifically enumerated GTest runs.

### Package the evidence

```bash
zip -r sdpa-2ch-acceptance-results.zip \
  sdpa-2ch-discovery sdpa-2ch-core sdpa-2ch-compat
```

Upload that ZIP, and the corresponding latest Thor snapshot if it differs from 738. Include any build failure output. Run a command on an RTX GPU; local static checks in the worker environment cannot prove CUDA execution.

## Performance evidence already collected (not a request for new benchmarking)

Matched public-layer Nsight runs on RTX 5090, B=16 uneven causal, Q=620 active / 2048 capacity, KV=872 active / 2048 capacity, 20 measured iterations, forward/training branches:

| Measured GPU kernel, median (µs) | Fixed | Neutral | Neutral / fixed |
|---|---:|---:|---:|
| FP16 inference forward | 5.248 | 4.960 | 0.945× |
| FP16 training forward | 5.168 | 5.152 | 0.997× |
| FP16 training backward | 8.864 | 8.864 | 1.000× |
| BF16 inference forward | 4.960 | 4.960 | 1.000× |
| BF16 training forward | 5.088 | 5.136 | 1.009× |
| BF16 training backward | 8.880 | 8.864 | 0.998× |

B=7 FP16 traces similarly confirmed matching generated attention kernels and nearly identical kernel durations. At B=16 cuDNN may choose different optimized forward variants; **further engine-selection analysis is out of scope**. These timings are kernel durations, not end-to-end submission or a guarantee about all workload sizes.

G4-C4→C7 host-allocation work in cuDNN wrapper, B=16 uneven causal BF16 neutral training, one forward + one backward:

| Metric | G4-C4 baseline | G4-C7 |
|---|---:|---:|
| `malloc` calls across both cuDNN wrappers | 149 | 53 |
| Ragged-metadata `malloc` calls | 60 | 0 |
| Process-wide `malloc` calls | ~889 | ~781 |

Thor-owned avoidable metadata and shape-validation allocations have been removed. The remaining `unordered_map` used by the cuDNN frontend is intentionally unchanged; do **not** redesign or cache it without an independent performance case. cuDNN internal host allocations cannot be ruled out. The G4 allocation measurements were from instrumented builds; perf comparisons above were GPU kernel traces.

## Final report after workstation gate

Record the two test-runner summaries (actual discovered, executed, passed, failed, skipped), compiler/build result, hardware/driver/build config, source diff relative to 724/738, perf evidence above, and limitations. **Do not claim external whole-network CUDA graph capture** or composite Attention 2D migration. Formal 2C acceptance is a CONTROL HUB decision after reviewing the user-run gate evidence.
