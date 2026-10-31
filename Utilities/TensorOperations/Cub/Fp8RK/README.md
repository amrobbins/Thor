# Retained FP8 RK VALUE reductions

This directory owns the retained original-input FP8 RK physical-kernel selector:
OneWindow, DeepRSmallK, AlignedK, and RuntimeK. The geometry-aware XGBoost
ranker and its fixed 28-candidate shortlist choose *first-stage physical*
kernels, not complete stamped reduction programs. The retained legal compiled
inventory is the authority for which candidates can be launched; unsupported
variants must not be synthesized as fallback kernels.

## Production selection and execution

`selectDensePhysicalPass()` is the ordinary production entry point for a
nominated R/KR/RK site. For eligible original-input FP8 RK VALUE it calls
`selectCachedIsolatedRKSite()` and converts its selected first-stage recipe
into `DenseRetainedFp8RKPhysicalPlan`. This is the **default** production route;
there is no environment-variable opt-in or old FP8 RK production fallback.
Original E4M3/E5M2 input supports all eight VALUE reductions (SUM, MEAN,
PRODUCT, MIN, MAX, L1Norm, L2Norm and SumSquares), with FP8, FP16, BF16,
FP32, and optionally FP64 final storage under Thor's ordinary build policy.
The typed retained selector handles the operation and input transform, and
Complete kernels perform finalization and runtime FP32 scaling before output
conversion. Staged kernels write unfinalized FP32 partials; subsequent
independently planned passes perform the finalization and scale exactly once.
Unsupported geometries still reject rather than invoke a legacy FP8
fallback. Other R/KR, ARG, view, and non-FP8 reduction paths have
independent dispatch.

The shared `planDenseReduction()` owns the **only** ordinary-dense VALUE
multi-pass planner. It selects one physical pass per considered site through
`selectDensePhysicalPass()`, compares the actual next-problem intermediate bytes
with its deterministic `smallestIntermediate` rule, advances the mathematical
problem, and repeats. A retained FP8 physical pass launches **one** kernel:

- A complete first stage writes the ordinary `[outer,1,inner]` result.
- A staged first stage writes a real **FP32** `[outer,shards,inner]` output,
  which becomes the next ordinary reduction problem with an outstanding
  `IntermediateReduction(shards)` axis. A later physical pass is selected
  independently by the global planner; it is **not** the FP32 pass chosen
  hypothetically during first-stage benchmarking.

Nonfinal mathematical outputs are FP32. Original-input transforms are applied
once; finalization, divisor and output conversion belong to the final
mathematical pass. The flat `DenseExecutablePlan` owns intermediate outputs and
per-pass physical scratch **per stamp**. `runOn()` launches those selected
passes without consulting the selector/cache or allocating/tuning workspace.
Scratch-free Thor kernels allocate no dummy scratch; CUB's two-phase APIs may
require a stamp-owned non-null minimum buffer even for a zero-byte query.

`CubReduction::queryWorkspaceSizeInBytes()` selects the same physical plan
as stamping and sums the exact nonfinal FP32 output allocations plus physical
scratch (with checked addition). The stamped executor repeats that accounting
against the buffers it owns. `getDensePhysicalPassDiagnostics()` exposes the
actual ordered pass families, input/output tensor geometries and dtypes, and
intermediate/scratch bytes; family-specific packet/CTA/shard parameters are
optional when they do not apply. There is no synthetic all-family RK geometry.

## Kernel work mapping

DeepRSmallK, AlignedK (packet, hoisted staged and unified complete) and
RuntimeK rotated divide nothing by a runtime work-space extent on the device.
They launch `reductionWorkGrid(shards, component_tiles, outer)`: shard on
`blockIdx.x` (kept fastest so resident blocks stream consecutive rows of one
outer slab), component tile on `y`, outer on `z`. Each axis is a 32-bit stride
loop, so geometries beyond the 2^31-1 / 65535 grid limits stay correct;
normally every block owns one work item. Every extent is limited to 2^31-1 so a
32-bit index plus its stride cannot wrap. AlignedK's balanced row split
(`ReductionShardRows`) is computed once on the host, and every shard's row
count must fit a 32-bit counter. Row loops use 32-bit counters and a 64-bit row
pointer; DeepRSmallK forms one 64-bit shard base pointer and uses 32-bit
offsets inside the shard.

Two measured exceptions keep the pre-P5-B persistent 65535-block grid, where
one block per work item was slower because launch and teardown outweigh the
loads: AlignedK Complete items of at most 8 rows and 512 bytes (11%), and
DeepRSmallK shards of at most 256 bytes (15%). Every other measured stage was
neutral or faster with a full grid.

The FP8 PrefixReserved rotated kernel is pinned to three 512-thread blocks per
SM (`__launch_bounds__(512, 3)`, i.e. at most 40 registers): at 41 or more
registers it drops to two blocks and loses 15-23% on one-tile shapes. Check
`cuobjdump -res-usage` for REG <= 40 and no LOCAL/STACK spill after changing it.

AlignedK lanes walk their rows with an explicit load-ahead
(`accumulateKParallelFp8Rows`): batches of 4 rows (P8) or 8 rows (P2) are
software-pipelined so the next batch is in flight while the current one is
decoded, and FP8 bytes are decoded in pairs. Accumulation order per component
is unchanged, so results are bit-identical to the plain row loop.

DeepRSmallK component-owned kernels with two or more components per lane
(K > 32) prefetch the packet of their next row group into a register before
decoding the current group from shared memory (`narrowFp8GroupWindow`), so one
group's global load is always in flight (0.98x to 0.89x time, best at 8
components per lane). One-component-per-lane and tiny-K kernels keep the plain
group loop, where prefetching measured 1-5% slower. Paired FP8 decode is not used
there: a lane owns one component, so its bytes sit K apart in shared memory and
packing them costs about as many instructions as the saved conversions.

DeepRSmallK's shard depth (`rows_per_shard`, one of `Fp8DeepRSmallKPolicy::compiled_rows`) is a runtime kernel
operand (P5-C), passed as log2 so the shard origin stays a shift: a runtime 64-bit multiply cost 2-4% on shards of
4 KB or less (1079). It only sets each shard's first row and clamps its row count; the row loops already used runtime
`rows_per_warp_group` / `shard_rows`, and the five compiled depths had identical SASS apart from that constant. One
kernel per (dtype, combine, transform, K ownership) now serves all five depths, except the one-component-per-lane
kernel (K 9..32): with the runtime depth it still lost 4.8% at rows64 and 1.9% at rows128 after the shift (1080),
so it is compiled at each depth again (`FixedRowsPerShardLog2`). 264 kernels instead of 840. The
selector still chooses the depth, and the launchers and occupancy queries validate it. The tiny-K kernels sit at
exactly 40 registers (6 blocks of 256 threads per SM); check `cuobjdump -res-usage` after changing them.

RuntimeK packet-complete keeps its persistent grid (several W-warp outputs
share a block) and its original per-work-item split; replacing that split with
a multiply-high division measured 4-5% slower in the R loop. OneWindow already
used 32-bit indexing with compile-time divisors and is unchanged.

## Complete epilogue (P5-A)

Complete kernels finalize through `finalizeFp32Epilogue()`. nvcc moves the launch-uniform output-dtype switch
outside unrolled store loops, so an inline `AdditiveFinalizeFp32` used to be copied once per store slot and output
dtype, each copy with its own IEEE divide and sqrt fast paths and per-kernel copies of the divide/sqrt slowpath
subroutines (48 copies, about 31 KB of a 40 KB AlignedK P8/CTA64 Complete kernel). Now:

- The identity case (`divisor == 1`, no square root) is one inline uniform test; MEAN and the square-root operations
  call `finalizeAdditiveFp32OutOfLine()`, one noinline copy per translation unit (the Cub objects are not
  relocatable device code).
- OneWindow and RuntimeK packet-complete compile no finalizer for PRODUCT/MIN/MAX (`FixedOp != 0`): the host only
  gives those the identity finalizer.
- The typed AlignedK packet kernel finalizes `local[]` once before its output-dtype switch, like the unified kernel.
  The unified kernel's Joined (P2/CTA32) placement uses the shared finalizer; its InBranch (P8/CTA32, P8/CTA64)
  placement keeps the inline finalizer, because the split raised P8/CTA32 from 79 to 82 registers and slowed
  identity-transform shapes by up to 48%.

Stage 1 removed about 9 MB of selected first-stage SASS (OneWindow 20.4 to 12.4 MB, device helpers 2.4 to 0.8 MB).
Results are bit-identical (same arithmetic, same order).

The output-dtype store stays inline at every store slot. Measured and rejected (1077): one noinline
finalize/scale/store for OneWindow kernels with 4 or more store slots per lane and for RuntimeK packet-complete. It
saved 1.6 MB, but RuntimeK SUM kernels went from 62 to 80 registers and lost up to 1.9x (oracle up to 29% on
the shapes they win), and OneWindow lost 0.7% (worst 4%) on the shapes those kernels win, with 42 kernels losing a
block per SM. After changing this, compare
`cuobjdump -res-usage` (REG, STACK) for the Complete kernels: a call can raise register counts or add stack.

## FP8 MIN/MAX fold (P5-F)

FP8 MIN/MAX compile no ABS/SQUARE variants in OneWindow and the RuntimeK rotated first stage, and OneWindow and
rotated PrefixReserved MAX launch the MIN kernel (PTX `min.NaN.f32`). Selectors and candidates are unchanged; a folded
request reads the occupancy of the kernel it launches. The input transform and the MAX sign are applied to the packed
FP8 bytes before decode, one LOP3 per four values (`Fp8ExtremumFold`, `CubReductionOperation.cuh`):

| | MIN | MAX |
|---|---|---|
| identity | `bytes` | `bytes ^ 0x80` (max x = -min(-x)) |
| ABS | `bytes & 0x7f` | `(bytes & 0x7f) ^ 0x80` (max \|x\| = -min(-\|x\|)) |
| SQUARE | as ABS, square the result | as ABS, square the result |

x^2 is monotone in |x| and FP8 squares are exact in FP32, so (max |x|)^2 equals max(x^2) bit for bit. An FP8 sign flip
keeps NaN as NaN and swaps +-Inf and +-0; `min.NaN` orders -0 < +0, so results, NaN propagation and signed zeros match
separate MAX kernels. The fold is undone once per result:

- OneWindow (Complete) squares in-kernel; the host negates `output_scale` for MAX (not needed for SQUARE).
- The rotated first stage negates or squares each FP32 partial before storing it, so staged partials (and the FP32
  continuation) are exactly those of the requested operation.
- Rotated Exact keeps separate MIN and MAX kernels and folds only the transform (`SignFold == false`: one AND mask,
  no XOR operand), pinned to two blocks per SM. It compiles at exactly 64 registers; the full fold took it to 67
  (one block per SM) and made it 1.47x slower (1084).
- RuntimeK packet-complete is not folded: with compile-time transforms already, the fold only added work and cost
  1-2% (oracle 1.011, worst 8%, on the MIN/MAX shapes it wins) for 10 kernels (1084, reverted in 1085).

Kernel counts: OneWindow 1008 to 756 (1082). RuntimeK 312 to 204 (1085): rotated MIN/MAX 144 to 36 (PrefixReserved
12, Exact 24), packet unchanged. SUM and PRODUCT are unchanged (they ignore the fold operand). Measured: OneWindow
MIN/MAX 1.5% faster overall (the runtime transform branch left the R loop; O1 at K 127-256 +0.2-0.8%); rotated
PrefixReserved MIN/MAX 0.997 (1084).

### Runtime FP8 format (rejected, 1086)

OneWindow keeps the FP8 format as a template axis. Measured (1086 vs 1085): a format-agnostic decode, with E5M2 FP16x2
bits from one byte permute, the E4M3 hardware FP8x2 -> FP16x2 conversion, and a byte permute with a launch-uniform
selector picking one (data flow, exact for every encoding), made OneWindow 1.5-2.3% slower in both formats (K >= 255
+4.5%, p95 1.37) and 1.35% slower on the shapes OneWindow wins (oracle, about 480 shapes per format > 2% slower).
14 kernels per format lost a block per SM (mostly 40 -> 44 registers; K512 SUM 64 -> 66). E5M2 gained nothing from
dropping its hardware conversion, so the conversion is not the bottleneck; the extra integer work and registers are.
A pure-integer E4M3 decode would cost more (E4M3 NaN 0x7f shifts to the finite FP16 1.875, x256 = 480, and needs a
per-value fix-up). It would have saved about 4.5 MB (756 -> 378 kernels); not worth these slowdowns.

## Load-ahead (P5-H)

Group and row order are unchanged, so results are bit-identical. Measured 1089 (everything pipelined) vs 1085:

- RuntimeK packet-complete: the row loop is software-pipelined (the next row's packet is loaded before the current
  row is staged and decoded) and pinned to four 256-thread blocks per SM (`__launch_bounds__(256, 4)`). It compiles
  at 56 registers or fewer (SUM was 62), no spills, and runs 0.79x on every operation; best configuration on the
  shapes it wins 0.877.
- AlignedK Staged hoisted (Identity) P8/CTA64: load-ahead batch 2 rows instead of 4
  (`K_PARALLEL_FP8_HOISTED_STAGED_LOAD_AHEAD_ROWS`), 0.944x. The typed ABS/SQUARE Staged P8/CTA64 kernels were
  1.04-1.05x with 2 rows and keep 4.
- OneWindow (the compile-time-K kernels, removed in P5-J2; the bucketed units fix their loop form per unit): the
  group loop can be pipelined the same way (`loadOneWindowPacket`, `stageOneWindowPacket`). Pipelining every O1 and multi-component kernel made them 0.91x overall, but it is a latency
  fix and slowed bandwidth-bound shapes, often the ones a kernel wins: O1 K 3-8 and 31 by 2-8%, O2 K 17/19 and
  O4/P16 K 9/17 by up to 14%, O8/P4 at every K; most other O2/O4/O8 classes never win a shape either way. The policy
  `ONE_WINDOW_LOAD_AHEAD` keeps only the classes that improve the best-configuration-per-shape result with no shape
  slower than about 2%: O1/P16 K 63 and 127-512, O2/P16 K 32/64, O4/P8 K 17, O4/P16 K 16/64, O8/P16 K 16. They run
  0.81x (O1 K 127-497 about 0.7x); best configuration on the shapes OneWindow wins 0.996 (284 shapes > 2% faster,
  1 > 2% slower). O1 K 16 and 32 were net positive but up to 3.3% slower on some winning shapes and stay plain.

## OneWindow is the bucketed runtime-K kernel (P5-I, P5-J)

Since P5-J2 (1098) the OneWindow family has one kernel, `fp8OneWindowRuntimeKCompleteKernel`: the former
compile-time-K OneWindow algorithm with K as a runtime operand. The 756 compile-time-K kernels (25 K values,
9,076,992 bytes of the 15,553,920-byte selected first-stage SASS) are gone. Reasons: production models use K values
outside those 25, where small-R shapes fell to kernels built for other regimes (AlignedK P2, RuntimeK packet-complete
on a mostly idle 33-byte tile) at 0.15-0.9 TB/s; and the bucketed kernel matched the compiled kernels on the shapes
they won.

- Selector mapping (`launchFp8GeneralRParallelRCooperativeCalibrationComplete`,
  `queryFp8GeneralRParallelRCooperativeCalibrationOccupancy`, unchanged names): the frozen variant (O, W, P) runs
  the unit for (O, P, components-per-lane bucket of K). Only P16 has units; the P4/P8 variants and the (ownership,
  bucket) pairs without a unit throw, which the selector treats as absent (residency 0).
  `Fp8OneWindowPolicy::oneWindowCoversK` (K <= 497, or 512) replaces `compiledK`. The ten variants, the ranker and
  the 28-candidate shortlist are unchanged until the ARG-aware retrain; OneWindow candidates now also appear at K
  outside the old 25.
- Units (`ONE_WINDOW_RUNTIME_K_UNITS`): P16 O1 c1, c2-la, c4, c8-la, c16-la; O2 c1, c2, c4, c8-la; O4 c2, c4, c8-la;
  O8 c1, c2, c4. One loop form per (O, P, bucket), so a variant maps to at most one kernel. 90 kernels (x 3 physical
  operations x 2 FP8 formats). Chosen by greedy oracle pruning over the 1094 focused and 1095 uncovered-K sweeps with
  no compile-time-K kernel present (geometric mean <= 1.004 and per-cell <= 1.10 against all 36 P16 units).
- Packet width and outputs per warp (1/2/4/8) are compile-time, warps per output runtime (O1). K is bucketed by
  maximum components per lane (1, 2, 4, 8, 16): `local[]` and the slot loop stay compile-time, and unused slots are
  skipped by the `owned_component < K` guard.
- The host computes what compile-time K provided (`Fp8OneWindowRuntimeKGeometry`: rows per staged group, R workers
  per component, accumulating lanes). The gather walks a row byte offset with lane-independent increments
  (workers x K); the one-component lane-to-(component, worker) split is one division per thread, and the worker fold
  is a short runtime loop once per output.
- Decode packing (P5-I2, after 1092): buckets 1 and 2 pack four rows of one component per FP8x4 decode, with
  unpredicated full steps and one tail; buckets 4-16 pack four of a lane's components of one row per decode. The
  first build packed rows in every bucket and lost 1.14x (c1) to 3.6x (c16) to the compiled kernels (a predicated
  four-row gather per slot for one valid byte at large K).
- Measured, 1097 `--focus=replacement` (293 shapes: OneWindow-relevant grid, 148 uncovered-K, 11 at K 2; E4M3,
  SUM/PRODUCT/MAX/SUM-square, 24/96/192 MiB), oracle without compile-time-K kernels vs with them:
  - Compiled-K shapes: 0.995 geometric mean; 56 of 1025 cells > 5% slower. Losses: K 2 at R 4-8 (up to 1.27x; the
    compiled O8 kernels at P4, P8 and P16 are all faster than the bucketed O8 c1, so the cost is per-output overhead
    at 8-16 bytes per output, not packet width), K 129 at R 40-64 (1.15x, MAX), K 17 at R 32 (1.10x, compiled O4 P8),
    K 4 at R 10 (1.08x). Faster at K 9, 11, 17, 19 (0.89-0.93, O1/W1 c1 rows-packed decode).
  - Uncovered K: 0.687 (by R*K: 0.52 at <= 256 bytes, 0.62 at <= 1 Ki, 0.84 at <= 4 Ki, 0.97 at <= 16 Ki, parity
    above). All cells: median input bandwidth 1.46 -> 1.50 TB/s, p10 0.47 -> 0.97, cells >= 1.3 TB/s 64% -> 79%.
  - The 8-byte packet (1097 built every ownership x bucket 1-8 with both loops): 1.0005 overall; of the
    selector-reachable ones (O4/O8) only O8 c1 helped, 8 cells by up to 6%. Not kept.
  - Registers: 38-48 for c1-c8, 56-64 for c16, no stack or local memory.
- Per-output overhead (P5-K1/K2, 1099-1100). The 1098 losses were small windows where the compile-time-K kernels
  folded constants: K 2 at R 4-8 and K 4 at R 10 (O8, one component per lane) and K 129 MAX (O1 c8).
  - Kept: one-component buckets fold the R workers of a component with butterfly shuffles when the worker count is a
    power of two (all of O4/O8; O1/O2 when lanes / K is a power of two, including one worker, which stores
    directly), with no shared-memory round trip and no warp barriers; other worker counts keep the shared fold. The
    SUM order of the worker fold is a tree (results can differ in the last bit from 1098; MIN/MAX/PRODUCT exact).
    1099 vs 1098: K 2 0.976 (worst cell vs the compile-time-K kernels 1.28x -> 1.19x); other compiled-K shapes
    neutral. K 4 at R 10 (one worker, already a direct store) did not move.
  - Rejected (1099-1100, reverted in 1101): 32-bit output element indices instead of the 64-bit multiply by runtime
    K. Computed before the R loop (1099) or after it (1100), they changed register allocation in the
    multi-component units: 1100 had O1 c16 SUM 63 -> 69 (over the 65 cliff; K 497 R 24 SUM 1.12x slower), O1 c8
    PRODUCT 48 -> 53 and O1 c2 MIN/MAX 40 -> 47. Since 1101 the multi-component units are source-identical to 1098
    and only the one-component worker fold differs.
  - Rejected (1099, reverted in 1100): buckets 4 and 8 decoded a lane's partial last group of four slots row-packed
    per valid slot when that took fewer FP8x4 conversions. OneWindow SASS 1.75 -> 3.84 MB, c8 units 48 -> 56
    registers (a block per SM lost), the c16 unit 64 -> 67-70 (with the early output index), uncovered K 33-111
    up to 1.18x slower (K 65 1.067), and K 129 unchanged (1.0045).
- History: 1092-1095 measured every P16 ownership x bucket 1-16 with both loops (216 kernels, 4.83 MB) next to the
  compiled kernels; 1096 kept 12 P16 units chosen with the compiled kernels still covering compiled K (72 kernels,
  1,055,488 bytes); 1097 added the 15-unit replacement set and the P8 measurement set (258 kernels). Through 1097 the
  test checked bit identity against the compiled kernels wherever one existed for that K, ownership and packet.
- Test: `Fp8OneWindowUnitsMatchReference` checks every unit against an exact CPU reference over a K list that covers
  all 15 units, checks that the selector-facing launcher runs the same unit, and that `fp8OneWindowRuntimeKUnits`
  agrees with `oneWindowCoversK` for K 1-520.

## Cache-miss measurement (not a production executor)

`Fp8RKSelectionCache` is a bounded, same-key single-flight cache of the
immutable `Fp8RKSelection` physical recipe only. It owns neither Tensors,
streams, scratch nor FP32 continuation descriptors. Cache hits do not choose
FP32 continuation passes. Cache clear/eviction cannot invalidate a stamp.

For each ranked FP8 first-stage candidate on a **cache miss**:

1. `planIsolatedRKSite()` describes an isolated benchmark of the nominated
   mathematical site, not a production planning step.
2. `makeSelectedIsolatedBenchmark()` asks `selectIsolatedFp32Continuation()`
   to choose each hypothetical FP32 pass, **sequentially** through the same
   official physical selector used by production. The candidate's temporary
   program includes all required FP32 intermediates and queried physical
   scratch within the measurement budget.
3. Only after every candidate's selection dependencies are resolved does the
   autotuner acquire the shared per-device GPU measurement mutex and compare
   complete candidate programs. The winner contributes **only its first FP8
   physical kernel** to the recipe cache; all benchmark programs die after
   measurement.

Device UUID, SM count and compute capability are queried once per process-local
GPU ordinal and reused on cache hits; a cache clear evicts recipes, not device
identity. Only a cache-miss leader allocates an internal representative Tensor.
Its allocation/fill and eventual `dropReference()` execute under the shared
per-device GPU measurement mutex so `cudaFree()` cannot overlap another tuning
contest. Dependency selection still occurs **without** that lock, and the
interleaved 28-candidate timing contest acquires it separately. A user-supplied
representative belongs to its caller and is never released by the selector.

The cache and GPU measurement locks have independent purposes: cache single-flight
coalesces the same key, while the device mutex serializes different-key GPU
measurement and internally owned representative setup/teardown. No mutex is
held while awaiting a same-key cache flight. A failed or cleared in-flight tune
cannot insert a stale result, and an existing stamped executor remains valid.


`Fp8RKSelectedContinuation{,Benchmark}` and `Fp8RKIsolatedSite` therefore stay
live **only to construct comparable autotune measurements**. They are not an
alternative production executor or a parallel FP32 physical policy.

## Validation

From the existing build directory:

```bash
cmake --build . --target Thor \
    thor_fp8_rk_isolated_site_smoke \
    thor_fp8_rk_selected_continuation_smoke \
    thor_fp8_rk_selected_continuation_gpu_smoke \
    thor_fp8_rk_stamp_autotuner_smoke \
    thor_fp8_rk_stamp_lifecycle_smoke \
    thor_fp8_rk_production_cutover_smoke \
    thor_dense_retained_fp8_rk_execution_smoke \
    thor_fp8_rk_selection_cache_smoke -j

./thor_fp8_rk_isolated_site_smoke
./thor_fp8_rk_selected_continuation_smoke
./thor_fp8_rk_selected_continuation_gpu_smoke
./thor_fp8_rk_stamp_autotuner_smoke
./thor_fp8_rk_stamp_lifecycle_smoke
./thor_fp8_rk_production_cutover_smoke
./thor_dense_retained_fp8_rk_execution_smoke
./thor_fp8_rk_selection_cache_smoke
```

`thor_fp8_rk_stamp_autotuner_smoke` exercises the real benchmark builder and
then the independent ordinary production stamp. The selected-continuation
smokes cover hypothetical zero/one/multiple FP32 dependencies. Lifecycle and
cutover smokes cover cache reuse, stamp independence, and production routing;
run them on an FP8-capable GPU. Review actual test outputs; building a target
alone does not establish GPU correctness or performance.

The isolated selected-inventory target (`thor_fp8_selected_inventory_size`)
reports only selector-reachable first-stage CUDA code, **not** total production
Cub SASS. Build `Thor`, then independently inventory the real Cub CUDA objects:

```bash
# From the existing build directory:
cmake --build . --target Thor thor_fp8_selected_inventory_size -j
python3 ../benchmarks/fp8_production_object_census.py \
    --build . --out fp8-production-census
```

Review `fp8-production-census/device-functions.csv` for additional FP8 kernel
functions. Family-name counts are **not** proof of selector reachability: the
production objects can instantiate different template specializations with the
same family name. The independent census now compares exact mangled kernel
identities against `fp8-selected-inventory/report/device-functions.csv` and
writes `fp8-production-census/selector-identity-comparison.csv`. The report
separates additional production-only identities from repeated compiled instances
and missing selected identities, per GPU architecture. After updating the
inventory analyzer, regenerate its report (the CMake target above does this).
Production-only identity does not automatically mean dead code: trace its
host dispatch and other R/KR, ARG, and specialized-view users before deletion.
The census intentionally does **not** declare such kernels obsolete.
Production object SASS and selected-inventory SASS are not the same as linked
`libThor.so` size. This read-only audit does not affect production dispatch or
emit additional kernels.

The active production code is complete for the retained FP8 VALUE cutover;
future FP32/FP16/BF16 autotuners and any broader shared cache/policy framework
are independent design decisions, not required by the present stamp lifecycle.
