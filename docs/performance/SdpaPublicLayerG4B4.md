# SERIALIZATION-OWNERSHIP-2C-G4-B4 — matched BF16 benchmark preflight

**Baseline:** user's `ThorKernels729.zip` (G4-B3 audit accepted locally).
**Scope:** benchmark only; no production/cuDNN/serialization/source changes.

The benchmark already supports the B=16 full FP16 and imbalanced causal FP16
shapes. This patch adds real BF16 Q/K/V and label host tensors, BF16 logical
ports and output, BF16 RAW-loss expressions, and active BF16 output and
`dQ/dK/dV` checks, along with B=2 full and B=16 full/imbalanced causal BF16
cases. The fixed and neutral halves of each case continue to share an
identical query/KV partition and active input value generation; the only
intended difference is geometry ownership. Dense remains an informative
full-row control, not a stand-in for uneven ragged work.

**Do not interpret a BF16 case name as evidence until it actually executes.**
A future pass through CMake and real GPU execution is required; this patch was
prepared on a host without CUDA. The existing CPU-reference numerical
regressions remain separate from the benchmark's fixed/neutral cross-check.

## Focused acceptance gate — from the repository root

```bash
# Apply the separate G4-B4 patch first.
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --list
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --case b2_full_bf16 --gpu 0
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --case b16_full_bf16 --gpu 0
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --case b16_uneven_causal_bf16 --gpu 0
./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
  --measure --case b2_full_bf16 --gpu 0 \
  --warmup 3 --trials 1 --iterations 3 --pool 4 \
  --csv sdpa-g4b4-b2-bf16-short.csv
```

Expected: the `--list` output has eight distinct cases including three
`dtype=bf16` entries; the three smoke commands each show `GPU_SMOKE_PASS`
and `PUBLIC_SDPA_GPU_EXECUTED`; the short measurement prints four
`WARMUP_COMPLETE` lines and `PUBLIC_SDPA_CONNECTED_BACKWARD_MEASURED`, and
produces **12 non-header CSV rows**, four configurations (fixed/neutral ×
forward/connected training) × three iterations. `dtype` must be `bf16` on
every timing row. CUDA skips, placeholder data, zero-gradient backward, or
nonzero exit status do not count as passing. The `--measure` mode checks
nonzero active `dV` after a true connected training warmup; it does not compare
numerical gradients against a CPU oracle.

The FP16 regression control, if needed:

```bash
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --case b16_uneven_causal --gpu 0
```

## After this gate passes

Use the existing isolated, post-warmup Nsight profiling path and phase-audit
script on `b16_full_fp16`, `b16_uneven_causal`, `b16_full_bf16`, and
`b16_uneven_causal_bf16`; one mode and geometry per capture, with matched
per-case configurations and execution sequence. Raw SQLite traces and
submission timing must be preserved. The G4-B3 audit script is intentionally
strict and will fail if it encounters an unrecognized CUDA/cuDNN phase count
or an unmatched sequence; investigate rather than loosening it silently.

**Not yet established:** device-input pool rotation (only independent HOST
buffers are rotated here), per-phase complete device elapsed time, calibrated
stamp cost, engine-plan identity, hot-path allocation/synchronization/plan
ownership, robust across-run parity, and whole-network graph capture. The CUDA
completion window remains host-enqueue contaminated. No speed parity or final
2C acceptance is asserted by this case-inventory patch.
