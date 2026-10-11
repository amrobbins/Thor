# SDPA G4-C1: matched Nsight trace and captured hot-path API census

Baseline: `ThorKernels730.zip`. This is **diagnostic only**: no production SDPA code is modified.

The existing strict G4-B3 phase script checks byte-identical GPU kernel names, and
correctly rejects the B=16 captures when cuDNN's independently autotuned forward
tile changes. The new tool checks the **kernel category sequence and counts**,
reports actual differences in cuDNN-generated forward/backward kernel names, and
retains separate per-geometry timing medians, p95, and per-submission kernel sums.
A different cuDNN tile is not silently relabeled as identical or a failure of
neutral geometry.

## Reproduce from existing trace directories

From Thor's root, with `sdpa-g4b4-traces` unpacked:

```bash
python3 scripts/sdpa_nsys_hotpath_audit.py sdpa-g4b4-traces \
  --case b16_uneven_causal --case b16_uneven_causal_bf16 \
  --out-prefix sdpa-g4b4-traces/b16_hotpath
```

Expected: four `SDPA_DEVICE_PAIR` lines and
`PUBLIC_SDPA_HOTPATH_TRACE_AUDIT_PASS`; output is
`sdpa-g4b4-traces/b16_hotpath.summary.csv` (eight case/mode/geometry rows).
For older B=7 traces, invoke separately with `--case b7_uneven_fp16`.

This script **requires** all four SQLite+submission CSV pairs per case; no
missing, zero-launch, or mislabeled recording can masquerade as a passing audit.

## Scope and interpretation

- Input pairing: identical B, dtype, mask, semantic maxima, active Q/KV counts,
  and packed capacities. Not a dense-versus-ragged performance assertion.
- Counts: per iteration one cuDNN forward, optionally one cuDNN backward, and
  two/four canonical partition conversion kernels; compare entire phase sequence.
- Report: per-phase median and p95 GPU durations, captured memcpy count/bytes,
  CUDA API memcpy/synchronization/allocator call counts, host submission-window
  medians, and generated kernel names.
- A matching generated kernel name is **not** a proved matching cuDNN engine-plan
  ID. Engine-plan IDs and independent autotune variability are not captured.
- An observed zero `cudaMalloc`/`cudaFree` API count does not measure C++ heap
  allocations or guarantee absence of allocations outside capture. Captured
  stream/device synchronization calls need phase attribution before labeling
  them as SDPA inner-loop calls.
- `submitBatch` includes public network scheduling, host staging and output
  delivery. Its `cuda_completion_window_ms` includes host enqueue gaps and is
  not an isolated GPU SDPA-kernel timing.
- Source-audit issue for subsequent G4-C ownership gate: current
  `CudnnAttention.cpp` constructs local `unordered_map<int64_t, void*> pack`
  inside each `forward()`/`backward()` call. These per-call maps plausibly
  allocate host memory (nodes and buckets); this is shared by fixed and neutral.
  Confirm it with **actual host allocation instrumentation** before a targeted
  production correction. This script does not instrument C++ allocation.
- Do not claim external whole-network CUDA graph capture from these traces.

## Next ownership gate

G4-C must identify heap-allocation call sites and cuDNN selection/plan identity
for fixed and neutral separately, without timing host setup as GPU execution.
Do not modify cuDNN/Thor attention kernels or change plan selection to force
matching tiles on the basis of single captured runs.
