# G4-C3: cuDNN SDPA host-allocation owner attribution

Baseline: the user-tested `ThorKernels732.zip`; this patch is diagnostic only.

G4-C2 observed approximately 470 `malloc` calls per inference submission and
approximately 889 per training submission on the RTX 5090. Those are *process-wide*
counts, not evidence that all allocations belong to cuDNN. G4-C3 adds an
**opt-in**, owner-scoped count for the wrapper's actual forward/backward calls.

- Normal Thor builds retain **no callback or wrapper overhead**. The CMake option
  defaults to `OFF`; never enable it for production or wheel builds.
- When enabled, only `CudnnAttention.cpp` is recompiled with the diagnostic hooks.
- Each scoped interval begins just before binding the tensor-pointer map and
  ends after cuDNN graph execution. It includes ragged metadata conversion and
  the synchronous host-side work of launching the cuDNN operation, but does
  **not** include preceding descriptor validation, placement, host batch
  submission, network scheduling, or later asynchronous device completion.
- A TLS marker tags allocations on the thread actually executing the wrapper;
  counters aggregate all Thor worker threads. The process-wide measurements
  remain available in the same CSV for comparison.
- Only glibc malloc-family calls are counted; custom/mmap allocation paths and
  CUDA device allocations are outside this probe. This is not a CUDA event
  benchmark, and process-wide timings under the probe must not be used for
  performance conclusions.

## Build and short gate

From the Thor repository root (preserving the existing Release configuration):

```bash
cmake -S . -B build-sdpa-perf -DTHOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=ON
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
cc -std=c11 -O2 -Wall -Wextra -fPIC -shared \
   -o build-sdpa-perf/libsdpa_host_alloc_probe.so scripts/sdpa_host_alloc_probe.c

mkdir -p sdpa-g4c3-alloc
for mode in inference training; do
  for geometry in fixed neutral; do
    stem="sdpa-g4c3-alloc/b16_uneven_causal_bf16_${geometry}_${mode}"
    LD_PRELOAD="$PWD/build-sdpa-perf/libsdpa_host_alloc_probe.so" \
    ./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
       --measure --case b16_uneven_causal_bf16 --gpu 0 \
       --profile-geometry "$geometry" --profile-mode "$mode" \
       --audit-host-allocations --audit-cudnn-scopes \
       --warmup 5 --trials 1 --iterations 5 --pool 8 \
       --csv "$stem.csv" || exit 1
  done
done
```

Expect four files with **five data rows each**. Each row must have
`cudnn_forward_calls >= 1`; training rows must additionally have
`cudnn_backward_calls >= 1`. Absence of a scope is an explicit failure rather
than a misleading zero-allocation pass. Check the columns for scoped malloc,
calloc, realloc, free, aligned/posix allocations and requested bytes. Inference
should not execute a backward scope.

This is an attribution gate. If owner-scoped allocations are substantial,
inspect their call sites and discuss ownership before any allocation-removal
patch. A per-call `std::unordered_map<int64_t, void*>` in
`CudnnAttention.cpp` is a likely contributor, but the counts alone cannot
prove every scoped allocation originates from that map. Do not share mutable
binding maps across concurrent CUDA execution without owner/concurrency design.

**After this diagnostic run, disable the option:**

```bash
cmake -S . -B build-sdpa-perf -DTHOR_SDPA_CUDNN_ALLOC_OWNER_AUDIT=OFF
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
```
