# SERIALIZATION-OWNERSHIP-2C-G4-B1 — public-layer submission timings

**Baseline:** accepted user-run `ThorKernels725.zip` (G4-A). This is the
**first bounded G4-B patch**, not full G4-B performance acceptance.

G4-A GPU execution and equivalence passed on the RTX 5090 for the B=2 full,
B=7 uneven, and B=7 uneven causal cases. This patch extends the same public
`ScaledDotProductAttention` benchmark executable without editing any
production SDPA kernels, plans, descriptors or placement code.

## What is measured

A `--measure` run constructs separate historical fixed-ragged (`2.1.0`)
and placement-neutral ragged (`2.2.0`) public networks with **identical**
Q/K/V values, row offsets, heads, semantic maxima, mask and batch size.
Each is stamped once. The benchmark preconstructs independent CPU `Batch`
objects (`--pool`), warms the physical network, then collects **raw rows**
for both of these modes:

- `forward_inference`: public `submitBatch(..., isInferenceOnly=true)`.
- `fwd_plus_true_bwd`: public `submitBatch(..., isInferenceOnly=false)` with
  a genuine RAW MSE loss joined to the real SDPA forward output, backward
  through Q/K/V GradientRivets, and checked finite active gradients with
  nonzero dV. This is **not an artificial bprop-only invocation**.

CSV columns include batch size, active Q/KV counts, physical capacity,
mask/dtype, separated placement/stamp cost, warmup and trial/iteration,
`host_ms`, and `cuda_completion_window_ms` for every iteration.

**Interpretation — critical**: `host_ms` is **host submission + waiting for
processing and named outputs**, not isolated GPU work.
`cuda_completion_window_ms` uses a private CUDA timing stream, a start event
before public submission, then a wait on the placement's completion event
and an end event. That GPU-timestamped window **still includes host enqueue
gaps** and other graph/device operations; it is not a kernel-only device
latency. `fwd_plus_true_bwd` includes forward + RAW-loss + backward, and
its difference from `forward_inference` must **not** be labeled backward
latency (different graph and objective paths). Engine identity is `unknown`
until a cuDNN/Thor plan inspection is reported.

This patch does not promise pinned or fully device-resident rotating input
buffers. Pool rotation is across independent CPU source allocations.
The graph retains static physical buffers. Cache avoidance, separately timed
GPU forward/backward, BF16, engine/plan evidence, CUDA allocations/sync audit,
and complete G4 comparative interpretation remain **G4-B2/G4-C work**.

## Optimized build and smoke

```bash
cmake -S . -B build-sdpa-perf -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --list
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --smoke --gpu 0
```

## Short nonzero forward and actual backward measurement gate

```bash
./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
  --measure --case b2_full_fp16 --gpu 0 \
  --warmup 3 --trials 1 --iterations 3 --pool 4 \
  --csv sdpa-g4b1-short.csv
head -n 8 sdpa-g4b1-short.csv
```

Expected 12 data rows (two geometries, two measured modes, three
iterations); stderr must show **four** `WARMUP_COMPLETE` lines and
`PUBLIC_SDPA_CONNECTED_BACKWARD_MEASURED`. Any exception or zero dV
is a failing gate. No CUDA device results were generated in this worker
environment; actual executable correctness must be verified by the user.

## Preliminary repeatable end-to-end boundary sweep (not kernel parity)

```bash
./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
  --measure --all --gpu 0 --warmup 8 --trials 5 --iterations 40 --pool 8 \
  --csv sdpa-g4b1-raw.csv
```

The sweep emits 5 cases × 2 geometries × 2 modes × 5 trials × 40 iterations
= 4,000 raw data rows. Only run it after the short gate. For isolated
kernel timing, take a Nsight Systems CUDA trace and attribute actual
forward/backward kernels; **do not** use the CSV GPU window as a stand-in.
