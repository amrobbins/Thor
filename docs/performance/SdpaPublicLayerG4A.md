# SERIALIZATION-OWNERSHIP-2C-G4-A: public SDPA benchmark preflight

**Baseline:** ThorKernels724, SHA-256 `b7949379aaf9ae6eba18277bdf65f6b319dc1dd940e6267241b3bd48ef231198`.

**What the executable actually does:** Constructs and places Thor's *public*
`ScaledDotProductAttention` API layer in separate fixed-ragged (`2.1.0`),
placement-neutral ragged (`2.2.0`) and, for full-row cases only, dense
(`2.1.0`) networks. It binds genuine Q/KV partitioned `Batch` inputs,
runs `PlacedNetwork::inferLogical()` on the selected GPU and checks
nonempty, finite active output and fixed/neutral numerical agreement. It
also asserts that a stamped physical `CustomLayer` exists and that ragged
output storage uses the expected physical `[B*maxQ, Hq, D]` capacity.

**What G4-A explicitly does NOT do:** measure elapsed GPU execution time,
forward/backward throughput, isolated device work, engine equivalence,
stamp cost, host allocation overhead, CUDA graph capture, or hot-path
allocations. `inferLogical()` necessarily includes submission, host/device
staging, output handling and synchronization. Its smoke result must NEVER
be interpreted as proof of optimized throughput. Only FP16 forward cases
are executable in this first preflight; BF16 and true RAW-loss backward
measurements are G4-B. The larger cases in the matrix are runnable smoke
fixtures, not measured benchmarks.

## Build on the user's CUDA workstation

Use a **separate optimized CMake build**, not the default Debug `build/`:

```bash
cmake -S . -B build-sdpa-perf -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-sdpa-perf --target thor_sdpa_public_layer_benchmark -j 8
```

If an optimized CMake directory already exists, regenerate that directory
with `-DCMAKE_BUILD_TYPE=Release`; do not run a Debug executable and call
it a performance baseline. Build configuration and CUDA architectures are
inherited from the repository's CMake setup.

## Matrix sanity without a GPU

```bash
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --list
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --matrix-csv > sdpa-g4a-matrix.csv
```

Check that both B=2 and B=7 cases appear, active values are nonzero and
`Q_capacity = B*maxQ`, `KV_capacity = B*maxKV`. Only full rows get dense
controls; padded dense work is not equivalent for uneven rows.

## Real GPU-launch smoke (must *not* silently skip)

```bash
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --smoke --gpu 0
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --case b7_uneven_fp16 --gpu 0
./build-sdpa-perf/thor_sdpa_public_layer_benchmark --case b7_uneven_causal --gpu 0
```

`PUBLIC_SDPA_GPU_EXECUTED` plus `GPU_SMOKE_PASS` confirms actual GPU
execution through fixed/neutral public layers (and a dense contextual
control for full rows). Exit code 1 means failure, including no CUDA GPU,
uncaught errors, missing physical layer, wrong active offsets or invalid
active output. `--all` exercises both larger shape fixtures; it is **not**
a performance sweep.

For the existing numerical regression gate, in the usual repository
`build/` directory run the 39-case GTest filter from the worker handoff,
checking **actual executed** totals and CUDA skips; the G4-A executable
is additional preflight coverage, not a replacement.

## G4-B extension points and measurement caveats

A future measurement patch must use the same case matrix and **the same
actual Q/K/V inputs** to compare neutral vs fixed public layer paths.
It must avoid timing CPU input creation or `inferLogical()` and must not
pretend `submitBatch()` is already a GPU-only measurement. Connect the
true backward to a real forward/RAW-loss and use GPU events/trace data to
separate forward and backward. Establish enough independent input storage
to avoid an L2-resident cache artifact; rotate inputs and keep stamp,
allocation and output staging outside device steady-state timings. Report
engine/backend identity. Do not infer optimized parity from this preflight.
