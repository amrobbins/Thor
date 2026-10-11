# G4-C2: Host allocation observation after warmup

This is **measurement-only**, relative to accepted `ThorKernels731.zip`.
No changes to production attention code. The probe is a separate GLIBC-only `LD_PRELOAD` library.
No GPU device allocations, cuDNN engine identity, graph capture, or SDPA-local allocation attribution are proven by this measurement.

## Build (from `~/Thor`)

```bash
cc -std=c11 -O2 -fPIC -shared -o build-sdpa-perf/libsdpa_host_alloc_probe.so scripts/sdpa_host_alloc_probe.c
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
```

## Short real GPU allocation gate

Run each test in its own process, after 5 warmups; the counter is reset per timed iteration:

```bash
mkdir -p sdpa-g4c2-alloc
for mode in inference training; do
  for geometry in fixed neutral; do
    stem="sdpa-g4c2-alloc/b16_uneven_causal_bf16_${geometry}_${mode}"
    LD_PRELOAD="$PWD/build-sdpa-perf/libsdpa_host_alloc_probe.so" \
      ./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
        --measure --case b16_uneven_causal_bf16 --gpu 0 \
        --profile-geometry "$geometry" --profile-mode "$mode" \
        --audit-host-allocations --warmup 5 --trials 1 \
        --iterations 10 --pool 8 --csv "$stem.csv" || exit 1
  done
done
```

Four CSVs with 10 rows each are expected. Every row has `host_alloc_scope=process_all_threads`
and counters for malloc/calloc/realloc/free/posix_memalign/aligned_alloc and requested bytes.
For stable distributions, increase trials (e.g. 3), change order of the geometries, and run independently of Nsight.

## Attribution and limitations

- This is **all-process/all-thread** allocation activity between `submitBatch()` and observed completion, not allocation by SDPA alone. Public submission does host staging, output maps, and events. Treat nonzero results as a prompt to isolate owners, not proof that SDPA allocated.
- This instrument intercepts dynamically resolved GLIBC allocator calls. It cannot count custom pools, `mmap`, static/internal non-interposed allocator entry points, memory allocated by the CUDA driver, or all heap traffic; allocation *requested bytes* is not live heap growth. Some allocator implementations call GLIBC internals without hitting public interceptors.
- It creates overhead while enabled and is **not comparable to uninstrumented microseconds**. Do not draw performance conclusions from allocator-enabled timings.
- `CudnnAttention.cpp` creates an unreserved `std::unordered_map` per forward/backward invocation; this is a code-review ownership concern. Follow-up G4-C should isolate actual map/graph allocation counts and plan identities before proposing a correction. A general public-submission zero-allocation claim is **not** established.
- Fully external CUDA Graph capture is still unproven.
