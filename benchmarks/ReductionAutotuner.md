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

1. Allocate one owning input Tensor as a ring of 32-byte-aligned views with
   at least 128 bytes of unused padding after every view, including when the
   logical tensor byte size is already aligned. Output and scratch rings use
   this same layout. Maintain at least eight times the device-reported L2 capacity between reuse of the
   *same input view* (override `Options::cold_reuse_bytes` if needed).
   Clone the representative input into all ring slots in O(log slots)
   GPU-copy submissions. Allocate separate rotating output and optional
   scratch Tensor storage, reserving the largest candidate scratch requirement.
2. One throwaway warmup per candidate by default, in input order.
3. Five rounds: first input order, next four independently reshuffled and
   distinct from the previous round (unless only one candidate).
4. Queue every start event, complete callback, and stop event on **one CUDA
   stream**, with **no CPU synchronization between candidates or rounds**.
   Synchronize once after all timing submissions; GPU timing events isolate the
   candidate interval. Neither setup nor input copying is timed.
5. For each candidate, discard the fastest and slowest elapsed time. Convert
   the remaining three elapsed times to logical-input throughput and average
   those three speeds. Select the **highest trimmed-mean throughput**; exact
   ties select the earlier user candidate. Return all five per-round timings,
   trimmed mean latency, trimmed mean throughput, index and name of winner.

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
