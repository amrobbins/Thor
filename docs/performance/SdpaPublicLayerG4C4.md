# SERIALIZATION-OWNERSHIP-2C-G4-C4 — Scoped allocator attribution

**Based on `ThorKernels735.zip`; diagnostic-only, not a performance fix.**

The G4-C3 results on B=16 uneven causal BF16 show the same allocation counts
inside the cuDNN attention wrappers for historical fixed and neutral ragged
geometries: inference forward 71 malloc/2 calloc; connected training forward
72 malloc/2 calloc and backward 77 malloc/2 calloc. The counts do not show
where in the wrapper those allocations occur.

G4-C4 extends the existing `THOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=ON` + GLIBC
`LD_PRELOAD` counter with three **mutually exclusive stages** per pass:

- `binding_validation`: construction/population/destruction of the binding
  `unordered_map` **plus descriptor checks and other wrapper host logic**.
  Not a map-only measurement.
- `ragged_metadata`: calls to `prepareCanonicalRaggedMetadata` (the
  conversion/dispatch function, not the subsequent `seqLen` bindings).
- `executable`: `executeGraph`, including workspace-pointer check and
  the cuDNN frontend executable invocation. Host operations inside the cuDNN
  frontend are included; this does not identify an internal cuDNN sub-owner.

Stage counts and requested bytes are appended to existing CSV columns; the
benchmark requires each of their sums to equal the corresponding wrapper total
or fails. The normal build has none of the new stage calls. No kernels,
serialized schema, cuDNN engine selection, or workspace policy are modified.

## Build

From the repository root after applying the patch:

```bash
cmake -S . -B build-sdpa-perf -DTHOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=ON
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
cc -std=c11 -O2 -Wall -Wextra -fPIC -shared \
  -o build-sdpa-perf/libsdpa_host_alloc_probe.so scripts/sdpa_host_alloc_probe.c
```

## Short gated run

```bash
mkdir -p sdpa-g4c4-alloc
bash -e <<'BASH'
for mode in inference training; do
  for geometry in fixed neutral; do
    stem="sdpa-g4c4-alloc/b16_uneven_causal_bf16_${geometry}_${mode}"
    LD_PRELOAD="$PWD/build-sdpa-perf/libsdpa_host_alloc_probe.so" \
      ./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
      --measure --case b16_uneven_causal_bf16 --gpu 0 \
      --profile-geometry "$geometry" --profile-mode "$mode" \
      --audit-host-allocations --audit-cudnn-scopes \
      --warmup 5 --trials 1 --iterations 5 --pool 8 \
      --csv "$stem.csv"
  done
done
BASH
```

Expected: five data rows in each of four CSVs, one nonzero forward call
per row and one backward call for training. The new 24 stage columns should
be populated, and the run aborts if their counts do not sum exactly to the
corresponding cuDNN wrapper totals.

## Limits

- GLIBC/malloc-hook counts are not GPU memory allocations and should not be
  interpreted as CUDA graph capture proof. These counters use a scoped TLS
  marker for attribution to Thor's current calling thread.
- Host timings from the instrumented build include interception overhead.
  Continue using the optimized uninstrumented benchmark for actual throughput.
- Different plans/kernels and CUDNN engine IDs remain to be audited separately.
- Reconfigure `-DTHOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=OFF` after the diagnostic.
