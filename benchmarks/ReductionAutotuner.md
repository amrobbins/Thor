# Reusable cache-cold reduction autotuning (Thor1001)

`Utilities/TensorOperations/Cub/ReductionAutotuner.h` provides a **CUDA single-stream**
reduction autotuner, intentionally independent of FP8 candidate enumeration.
Pass executable callbacks rather than kernel function pointers, allowing a
candidate to launch either a complete reduction or the first stage plus a
required FP32 continuation / finalizer. All candidates must produce the same
logical output and have the same operation semantics. The selected result is
an **index and name** in the caller's original candidate vector, with detailed
per-candidate timings. It is not a cached executable plan yet.

## Timing contract

1. Allocate one owning input Tensor as a ring of 256-byte-aligned views (1163;
   cudaMalloc's alignment, so a timed input is aligned as a real tensor's) with
   at least 128 bytes of unused padding after every view, including when the
   logical tensor byte size is already aligned. Output and scratch rings use
   this same layout and the same slot count. Keep at least eight times the
   device-reported L2 capacity of traffic between reuses of the same slot,
   counting each slot's input + output + scratch bytes (1161; override
   `Options::cold_reuse_bytes` if needed), so every timed launch's first stage
   reads an L2-cold input. Clone the representative input into all ring slots
   in O(log slots) GPU-copy submissions.
2. One throwaway warmup per candidate by default, in input order.
3. **Probe** (1161 KR, 1162 every contest): one timed launch per candidate on
   the next cold slot, then one host synchronization to read them.
4. From the probes (`planFromProbes`): each candidate's launches per
   measurement = ceil(`probe_target_us` / probe), 1 to
   `max_launches_per_measurement` (defaults 200 us and 32), so a short kernel
   batches enough launches that the ~2 us event resolution is about 1%, and a
   slow one is not repeated. A candidate whose probe exceeds `abandon_ratio` x
   max(best probe, `abandon_floor_us`) (defaults 3 and 8 us) is not measured
   and cannot win; `abandon_ratio = 0` measures every candidate.
5. `measurement_count` rounds (FP8: 5 Quick, 7 Regular, 11 Ultra): first in
   input order, then independently reshuffled and distinct from the previous
   round. Each measurement is one event interval around the candidate's
   launches, each on the next cold slot, divided by the launch count. All
   rounds are queued on **one CUDA stream** with no host synchronization until
   the end. Neither setup nor input copying is timed.
6. For each measured candidate, discard the `trim_count` fastest and slowest
   samples and average the rest; the smallest trimmed mean wins, exact ties
   going to the earlier candidate. Every candidate's probe, launch count,
   samples and trimmed mean (or `abandoned`) are returned with the winner.

Why (1160/1161, 3258 FP8 KR sites): with one launch count per site, a stamp
took 4 ms plus its contest's GPU time, and below 16 MiB 92-98% of that went to
shapes 10-100x slower than the winner (median 200-440 ms per stamp). With this
policy the median stamp took 11-29 ms by size, with the same pick quality.

A smaller kernel launch latency corresponds to higher throughput for the
identical logical reduction input, so no implementation gets an artificial
bonus for issuing *more* scratch-memory traffic. Candidate callbacks MUST be
nonblocking; calling `cudaStreamSynchronize`, `cudaDeviceSynchronize`, etc.
inside them invalidates the single-barrier timing design. The caller must
ensure its device, execution stream, and any captured plan resources survive
until `autotune()` returns.

Only the input seed and supporting ring copies are enqueued before timings;
output and scratch need no preinitialization because kernels are required to
write all locations they subsequently read. Scratch is a byte Tensor with
at least the candidate's declared `scratch_bytes` capacity; only that prefix
may be touched. Zero scratch bytes yield an uninitialized scratch handle.

**Usage:**

```cpp
using namespace ThorImplementation::ReductionAutotuning;
std::vector<Candidate> plans;
plans.push_back({"complete", 0, [](const Tensor& input, Tensor& output,
                                      Tensor&, Stream& stream) {
    // Launch the COMPLETE reduction plan on stream.getStream().
}});
plans.push_back({"staged", scratch_bytes, [](const Tensor& input,
                                             Tensor& output, Tensor& scratch,
                                             Stream& stream) {
    // Launch first stage writing scratch, then launch FP32 continuation.
}});
auto result = autotune(plans, representative_input, output_descriptor, stream);
// plans[result.winner_index] or result.winner_name
```

This utility itself does **not** integrate production stamp-time caching,
select FP8 family variants, or change `run()`.

## Validation

```bash
# From ~/Thor:
g++ -std=c++23 -O2 -Wall -Wextra -Werror -I. \
  benchmarks/ReductionAutotunerCoreSmoke.cpp -o /tmp/reduction_autotuner_core_smoke
/tmp/reduction_autotuner_core_smoke

# From ~/Thor/build-release, on an NVIDIA GPU:
cmake --build . --target thor_reduction_autotuner_smoke -j 8
./thor_reduction_autotuner_smoke
```

The GPU smoke compares one complete reduction, one first-stage + FP32 copy
continuation, and a deliberately slower complete reduction. It prints the
selected name and trimmed-five samples' metrics (mean over surviving three).
