# SERIALIZATION-OWNERSHIP-2C-G4-B3 — kernel/host boundary trace audit

**Baseline:** `ThorKernels728.zip`, following a passing G4-B2 Nsight trace gate.

This is an offline, report-only diagnostic patch. It adds
`scripts/sdpa_nsys_phase_audit.py`. No C++ benchmark or production attention
code is modified, and no new GPU run is needed for the focused test below.

The `b7_uneven_fp16` traces captured 20 **actual** public-layer invocations
for each of fixed/neutral inference and fixed/neutral connected training on an
RTX 5090. They have identical generated cuDNN Flash Attention forward and
backward kernel names and launch sequences. Individual generated kernel
medians, not whole-pass device latency, are 1.888 us inference forward,
2.144 us training forward, and 4.064 us training backward for both geometries.

The traces also contain host staging, host/device copies, CUDA event calls,
other GPU kernels, and idle gaps. Thus the measured submission-to-completion
CUDA event window is **not** isolated SDPA execution time, and summing kernels
would not account for possible overlap. Fixed/neutral host medians for this
particular capture differ markedly in *opposite directions* between modes;
no causation or parity at the full submission boundary is claimed.

## Offline machine-readable gate

From the Thor root, after collecting the four G4-B2 `.sqlite` files and
`.submission.csv` files into `sdpa-g4b2-traces/`:

```bash
python3 scripts/sdpa_nsys_phase_audit.py sdpa-g4b2-traces \
  --case b7_uneven_fp16 --out-prefix sdpa-g4b2-traces/b7_phase_audit
```

Required success: two `MATCHED_DEVICE_KERNEL_SEQUENCE` lines (60 inference
and 260 training kernel launches), one `MATCHED_BACKWARD_KERNEL` line,
and `PUBLIC_SDPA_NSYS_PHASE_AUDIT_PASS`. Outputs:

- `b7_phase_audit.summary.csv` — per-capture fprop/bprop medians and p95,
  generated-kernel fingerprints, full kernel counts, host completion medians,
  aggregate device-copy counts/bytes, and selected CUDA API counts.
- `b7_phase_audit.kernel-samples.csv` — the raw generated fprop, bprop,
  and partition-conversion kernel durations (no invented samples).

The audit fails on missing/zero traces, missing cuDNN kernel evidence,
non-matching generated kernel sequences, or mismatched submission counts.
The GPU kernel fingerprints identify the compiled implementations; they are
**not** cuDNN backend engine-plan IDs. Host event synchronization and copies
belong in the subsequent source/trace ownership audit, separating public
submission orchestration from physical SDPA. Zero allocation calls found in
these captured CUDA API events is not proof that all allocations, planning,
other APIs, or outside-capture activity are absent.

## Open work

Before formal 2C performance acceptance: matched larger and imbalanced/causal
cases; BF16 support; more trials and stable repeated capture order; real GPU
input-pool rotation instead of just independent host buffers; cuDNN
engine-plan identity if available; hot-path owner audit (copy/sync/alloc/
workspace/descriptor/plan); and non-profiler timing validation. Whole-network
external CUDA graph capture remains unproven. Do not change the optimized
cuDNN kernels in response to these preliminary host submission differences.
