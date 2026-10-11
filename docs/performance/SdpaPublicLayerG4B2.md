# SERIALIZATION-OWNERSHIP-2C-G4-B2 — isolated kernel trace preflight

**Baseline:** locally user-tested `ThorKernels726.zip` (G4-B1).
This patch does **not** replace G4-B1 or claim full performance acceptance.

The uploaded B1 CSV contained 12/12 expected raw timing rows and a successful
connected-backward run on an RTX 5090. For B=2 FP16 full rows, host medians
(inference fixed/neutral) were 68.639/70.171 us; connected training
187.211/192.952 us. The CUDA completion-window medians were
56.384/54.560 us (inference) and 169.920/175.104 us (training). Both
windows include host-enqueue gaps; these are preliminary, not kernel times.

## What this patch adds

- `--profile-geometry fixed|neutral` and `--profile-mode inference|training`
  restrict the existing real public-layer benchmark to exactly one configuration
  per process. Omitting both retains the original four-config G4-B1 gate.
- `--capture-gpu-kernels` wraps **only the timed trials** in CUDA
  `cudaProfilerStart`/`cudaProfilerStop`. Placement, cuDNN plan selection,
  preconstructed CPU input pool, warmups and gradient verification happen
  before capture. Nsight Systems can collect actual GPU kernel activity from
  this tight, single-configuration region.
- `scripts/sdpa_nsys_kernel_rows.py` exports *raw per-GPU-kernel times* from
  Nsight Systems SQLite, refusing missing/empty GPU traces. **Kernel durations
  are not elapsed attention-pass durations** and should not be summed when
  kernels overlap. Connected training includes real forward, loss and backward;
  neither this script nor a bare kernel name constitutes verified attribution
  to exactly one phase. Inspect GPU/API correlations for that.
- No production code changes, cuDNN graph/plan rebuilding, new kernels, or
  benchmark-input allocation during timed iterations.

## Optimized build and quick compatibility gate

```bash
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --measure \
  --case b2_full_fp16 --profile-geometry neutral --profile-mode training \
  --warmup 3 --trials 1 --iterations 3 --pool 4 \
  --csv sdpa-g4b2-neutral-training.csv
```

Expected: one `WARMUP_COMPLETE`, one `PUBLIC_SDPA_CONNECTED_BACKWARD_MEASURED`,
and **3 CSV data rows**, no zero-gradient or GPU skip. This command DOES NOT
use `--capture-gpu-kernels`, and thus works without Nsight.

## Real GPU kernel trace gate (four independent profiles)

From the repository root on a GPU workstation with Nsight Systems (`nsys`):

```bash
mkdir -p sdpa-g4b2-traces
for mode in inference training; do
  for geometry in fixed neutral; do
    stem="sdpa-g4b2-traces/b7_uneven_fp16_${geometry}_${mode}"
    nsys profile --force-overwrite=true --trace=cuda,cudnn \
      --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output="$stem" \
      ./build-sdpa-perf/thor_sdpa_public_layer_benchmark \
      --measure --case b7_uneven_fp16 --gpu 0 \
      --profile-geometry "$geometry" --profile-mode "$mode" \
      --capture-gpu-kernels --warmup 5 --trials 1 --iterations 20 --pool 8 \
      --csv="${stem}.submission.csv"
    nsys export --type=sqlite --force-overwrite=true \
      --output="${stem}.sqlite" "${stem}.nsys-rep"
    python3 scripts/sdpa_nsys_kernel_rows.py "${stem}.sqlite" \
      --case b7_uneven_fp16 --geometry "$geometry" --mode "$mode" \
      --csv="${stem}.gpu-kernels.csv"
  done
done
```

**Passing trace preflight:** four `PUBLIC_SDPA_GPU_KERNEL_TRACE_CAPTURED`
messages, four submission CSVs with 20 data rows each, four SQLite files,
four raw GPU kernel CSVs, each with nonzero GPU launches, and no exceptions.
Inspect actual kernel launch activity as well as cuDNN calls; *do not infer*
which individual kernels belong to fprop/bprop from aggregate training-mode
latency alone. If `nsys export` isn't supported in your installed version,
report its `--help` output rather than fabricating timings.

Start with this smaller case. Only then repeat for `b16_full_fp16` and
`b16_uneven_causal`; active-token densities and padding differ, so never
compare unequal geometries as though they were identical. The RTX 5090
machine may require a shorter Nsight capture or larger input pool depending
on local workload.

**Limitations:** This patch traces actual device kernels but provides no
cuDNN engine identity or isolated SDPA forward/backward **pass** durations yet.
It does not identify per-pass GPU boundaries, guarantee an independent GPU
rotating input pool, support BF16, prove zero allocation/host synchronization,
or prove external full-network CUDA graph capture. Such claims are gated by
further G4-B/C source audit and experiments, not by these traces alone.
