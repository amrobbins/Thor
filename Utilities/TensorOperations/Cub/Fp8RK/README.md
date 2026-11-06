# Retained FP8 RK VALUE reductions

This directory owns the retained original-input FP8 RK physical-kernel selector:
OneWindow, DeepRSmallK, AlignedK, and RuntimeK. The geometry-aware XGBoost
ranker (1141: retrained on end-to-end cold-L2 sweeps, one model for VALUE and
ARG, `benchmarks/fp8_ranker/`) and the autotuned shortlist choose
*first-stage physical* kernels, not complete stamped reduction programs. The
shortlist depth is the reduction's `CubReductionTuningDepth`: Regular (default)
times the 12 highest-ranked candidates for 7 rounds, Quick (opt-in) the top 6
for 5 rounds, Ultra (opt-in, 1143) the top 24 for 11 rounds. Each candidate's
fastest and slowest rounds (Ultra: two of each) are dropped and the smallest
average wins. Every timed first stage starts L2-cold, and a staged
candidate's continuation reads its fresh partials as in a real run. The
in-process cache keys on the depth, and a deeper entry (Ultra > Regular > Quick)
answers a shallower request. The retained legal compiled
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
and FP32 final storage (FP64 reductions were removed in 1258 / 1259).
The typed retained selector handles the operation and input transform, and
Complete kernels perform finalization and runtime FP32 scaling before output
conversion. Staged kernels write unfinalized FP32 partials; subsequent
independently planned passes perform the finalization and scale exactly once.
Unsupported geometries still reject rather than invoke a legacy FP8
fallback. Other ARG, view, and non-FP8 reduction paths have independent
dispatch.

## Retained FP8 R / KR (1146-1148, `../Fp8KR/`)

Original-input FP8 R and KR sites (the reduced run is innermost, so each output
reduces one contiguous row) have their own retained family, launched by
`launchFp8KRValueSegments` / `launchFp8KRArgSegments`, in three sub-families:

- **Pack**: one thread owns several short rows (row <= 32 bytes, span <= 128
  bytes) and walks their bytes with a row counter.
- **Group**: G = 1..32 lanes per row, each lane reading every G-th word.
- **Split**: one CTA per chunk of a long row writes a partial to the pass's
  scratch; a finishing kernel (one CTA per row) combines them in chunk order.
  No atomics: 1147's per-row ticket saved one launch, which measured as nothing.

All read 4-byte words on the tensor's word grid and mask bytes outside their
work, so any input address works. VALUE compiles only reachable semantics: SUM x
{identity, |x|, x^2}, PRODUCT, and MIN / MAX through RK's shared MIN kernel with
the P5-F byte fold. `fp8KRCandidateShapes` orders the shapes by a geometric
first guess (Pack first for short rows, Split first for few long rows); the
tuning depth keeps its prefix (Quick 6, Regular 12, Ultra 24, with the RK
rounds and trim). 1148 removed the direct CUB pass from the candidates. Short
passes batch several launches per timed interval (each on its own cold ring
slot), because CUDA events here resolve about 2 us. Selections are cached per
process like RK's (deeper answers shallower).
`thor_fp8_dense_site_audit --candidates` times every retained shape per cell.
1151: `thor_fp8_kr_training_sweep` is the KR counterpart of the RK
`--training` sweep: seeded grid and log-uniform random [K, R] sites (pure R
included) with sampled semantics, a seeded shuffle so any `--max-cells` prefix
is representative, held-out row lengths, and every listed shape plus the legal
shapes around it timed interleaved with cold L2. `fp8_kr_training.py` separates
kernel gaps (no shape reaches the target) from list gaps (an unlisted shape
wins) and reports the regret of each tuning depth.
1152: the sweep times the RK sweep's way, which reproduces to about 3 digits:
every launch reads its own view of a large input ring and writes its own view of
an output ring (no flush kernel; outputs pay their write-back, which a shared
output hid at short rows), cells are submitted in asynchronous batches with
passes interleaved across them, and the production stamp's pick is recorded.
`fp8_kr_training.py --compare=` measures run-to-run agreement. The 1145-1150
audit's flush-per-sample timing disagreed with itself by up to 28%; use the
sweep for KR / R timing.
1153: each candidate's launches per sample come from its own probe (1152 spent
80% of its GPU time on shapes 5x or more off the cell's best), and a copy curve
timed the same way gives each cell the time a memcpy moving the same bytes takes
(`copy_us`), so small sizes can be judged against what is reachable there.
1154: the 1153 full sweep showed Split throughput rising to just under each
multiple of the partial kernel's resident CTAs (6 per SM on sm_120) and falling
by up to 20% just past it. `fp8KRCandidateShapes` now leads few-row Split with
chunk counts that make 2, 1, 3 and 4 whole waves
(`fp8KRSplitBlocksPerMultiprocessor()` gives the residency per device), and also
offers Split (8, 4, 16, 32 KiB chunks, interleaved with Group) for many rows of
16 KiB or more, where ARG Group lost 5-15% to it.
1155: the 1154 sweep refuted the wave reading (whole-wave chunks won 27 of 1048
cells and lost 2% on average), so they are gone; the long-row Split stays (list
gaps 74 -> 15). The real 16-64 MiB gap is a fixed cost: Split and long-row Group
stream at 1.66-1.72 TB/s but pay 3-4.6 us per operation (memcpy: under 0.5 us),
and 1.5 TB/s at 16 MiB allows about 1 us. `--split-stages` times each Split
shape's partial kernel alone, so `fp8_kr_training.py` reports the finishing
kernel's share of that cost.
1155 result: the Split partial kernel alone runs at memcpy speed (0.99-1.02x a
copy of the same bytes); the finishing launch costs 1.3-1.4 us and is the whole
Split gap at 16-64 MiB (1.42-1.46 TB/s full, 1.52-1.55 without it). Removing it
on sm80 means in-kernel completion (an atomic ticket), a decision pending.
1156: a Block family (one CTA per whole row, direct store, no scratch) for many
rows of at least 1 KiB, listed second from 4 KiB rows: where Group had under 4
CTAs per SM (1.32-1.39 TB/s at 16-64 MiB) the same mapping ran 6-8% faster as
the Split partial kernel.
1156 result: Block won 147-197 of the ~225 cells per kind in each 1K-64K row
bucket; 16-64 MiB at 1.5 TB/s rose from 56% to 60% (VALUE) and 32% to 36%
(ARG), and ARG R 1K-64K's fixed cost fell from 4.6 to 3.1 us. 1157: Block is
listed first for rows of 2-64 KiB with at least one row per SM (where it won
81-100% of cells), and nowhere else (Group wins below 2 KiB, Split from 64 KiB).
1158: the RK selection setup for KR. `fp8KRCandidatePool` (the heuristic list,
then the legal shapes around it) is what the training sweep times and what
production ranks; `fp8KRRankerFeatures` (24) are written by the sweep beside each
time; `benchmarks/fp8_kr_ranker/train.py` trains the XGBoost ranker and exports
`Fp8KRRankerModel.h`; `fp8KRRankedShapes` gives the autotuner the top
`FP8_KR_SHORTLIST_DEPTHS` of the ranked pool. The placeholder model is untrained,
so until a trained header is installed the autotuner times the heuristic list's
prefix as before. A preview on the 1156 data (gradient-boosted regression,
geometry-separated folds) put the best shape within the top 3 to within 3% (max)
on held-out row lengths, against 13-134% for the list's top 3.
1159: the trained KR ranker is installed (12256 cells, 121274 timed candidates,
300 trees, model SHA256 c82f0c6b...; header parity vs XGBoost 2.4e-7 over 256
fixtures). Out-of-fold, the model's top 2 is within 1.2% (VALUE) / 1.8% (ARG) at
p99 and its top 8 holds the oracle shape in every cell; the list's top 2 was
within 183-226% at p99 and needed all 12. 1159 set
`FP8_KR_SHORTLIST_DEPTHS` to {2, 4, 8}; 1160 sets {6, 12, 12} (Quick, Regular,
Ultra), the RK candidate counts at Quick and Regular, for margin on GPUs and sites
the sweep did not see (out-of-fold, the top 8 already held the oracle in every
cell). Pools hold at most 15 shapes, so Ultra stays at 12 and differs from
Regular by rounds and trim, which still come from `fp8RKTuningPlan`.
1160 confirmation (fresh seed, 3258 cells): the ranker's top 8 held the fastest
shape in every cell; production picks were 0% over it on average, 4% at worst.
A stamp's wall time was 4 ms plus its contest's GPU time (r = 0.9998), and below
16 MiB 92-98% of that went to shapes 10-100x slower than the winner, each run
32 launches per measurement (median 200-440 ms per stamp). 1161: KR contests are
probed (`ReductionAutotuning::Options::probe_target_us`): one timed cold launch
per candidate sets its own launches per measurement (about 200 us each, at most
32), and a candidate over 3x max(best probe, 8 us) is not measured. Replayed on
the 1160 data this keeps every cell's best shape and drops no shape within 2%;
the median contest falls to about 9 ms below 16 MiB and 24 ms from 128 MiB. The
cold rings now count a slot's input + output + scratch bytes toward the 8x L2
reuse distance (it was input bytes alone, with the output and scratch rings at
the same slot count: up to GiBs of output ring for small inputs); this applies
to RK contests too, which are otherwise unchanged (one launch per measurement).
1161 GPU (same seed): median stamp 405 -> 11 ms below 1 MiB, 225 -> 12 ms at
1-16 MiB, 82 -> 19 ms at 16-128 MiB, 49 -> 29 ms from 128 MiB; picks as good.
1162: one contest policy for every reduction autotune. The probe, per-candidate
launch counts and abandonment are `ReductionAutotuning::Options` defaults, so RK
VALUE and ARG contests use them as KR does (callers set only rounds and trim).
For RK this also replaces one launch per measurement: short RK kernels were
timed with a single launch against the ~2 us event resolution. Audit:
`thor_fp8_retained_cold_l2_sweep --training --stamp` and `fp8_rk_stamp.py`.
1164: the RK training sweep times as the KR sweep does (`benchmarks/ColdTiming.h`:
probe-based launches per ~1 ms sample, rotating outputs). On the 800-cell stamp
audit the picked shape's sample spread fell from 8-22% (below 16 MiB) to 0.5-1.3%,
and RK picks measured 0.0-0.4% over the fastest shape on average (95-100% within
2%): the earlier RK "losses" were the one-launch sweep, not the stamp.
1165: the RK geometry ranker is retrained on 1164-method data: 15,614 cells at
1-256 MiB (seed 1165, 6,000 random shapes), 242,992 timed candidates, 300 trees,
model SHA256 b1a7bc53..., parity vs XGBoost 1.5e-9 (256 fixtures, bit-exact in
C++). It replaces the 20,452-cell model trained on one-launch samples at 96/192 MiB.
Out-of-fold, worst regret of the top N (ARG / VALUE): N=1 1.77 / 1.35 (was 331 /
5.1 with the frozen model's leaders-first order); N=6 1.077 / 1.040 (was 1.32 /
1.31); N=12 1.017 / 1.015 (was 1.07 / 1.14); p99 1.000 from N=6. Held out: N=6 max
1.033 / 1.010, VALUE exact from N=12. The trainer's depths were Quick 2, Regular 4;
`fp8RKTuningPlan` keeps 6 / 12 / 24, as KR keeps 6 / 12.

The shared `planDenseReduction()` owns the **only** ordinary-dense VALUE
multi-pass planner. It selects one physical pass per considered site through
`selectDensePhysicalPass()`, compares the actual next-problem intermediate bytes
with its deterministic `smallestIntermediate` rule, advances the mathematical
problem, and repeats. 1167 (user policy): sites are considered lazily, best
bound first. A site's bound is the bytes left if one pass reduced it fully;
sites are visited in (bound, KR before RK, outer before inner) order, and a site
is selected (for FP8, autotuned) only while its bound could still beat the best
actual candidate so far. A pass that completes its site therefore ends the step
after one contest; a staged pass leaves more bytes and lets the next site be
tried if its bound is smaller. The plan equals selecting every site (checked on
600 random problems with scripted selections); only the number of contests
changes. The finished plan is stamped as a whole. A retained FP8 physical pass
launches **one** kernel:

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

## FP8 ARG keys (ARG-A1, in progress)

ARG (argmin/argmax) for original FP8 RK input is being built on `../Fp8ArgKey.h`. No production kernel uses it yet;
today's FP8 ARG still runs the FP32-candidate tiled ARG kernels (1102 baseline: 3.87x slower than VALUE MIN/MAX).

- Each FP8 byte maps to an 8-bit key whose unsigned order is the value order: positive `x ^ 0x80`, negative
  `x ^ 0xFF`, -0 merged into +0's key 0x80, NaN key 0. `fp8ArgKeys4()` does four bytes per 32-bit word with plain
  integer ops (no decode, no per-byte video intrinsics).
- Argmax is argmin of the sign-flipped value (`xor_mask` 0x80808080) and the NaN set is one per-byte bias
  (E4M3 1, E5M2 3), both runtime: one kernel per geometry covers argmin/argmax x E4M3/E5M2. Unlike the rejected
  runtime FP8 format for VALUE (1086), there is no decode to make format-dependent.
- Lane state is one `uint32` per component, `(key << 24) | (row << 1) | negative_zero`, reduced with one unsigned
  min: lowest key, then lowest row. That is the existing ARG contract (NaN propagates, lowest index wins ties,
  -0 ties +0) with the same register count as an FP32 accumulator. A lane covers fewer than 2^23 rows (kernel
  precondition). Lanes widen to `uint64` `(key << 33) | (index << 1) | negative_zero` and merge with the same min.
- The winning value is rebuilt from the key and the negative-zero bit (`fp8ArgValueByte`), exact for every non-NaN
  byte, with no re-read of the input.

Tests: `Fp8ArgKey.*` in `test/Utilities/TensorOperations/Fp8ArgKeyTest.cpp` check every ordered pair of encodings for
both formats and operations against the FP32 ARG rules (`Fp8ArgReference.h`), packed vs byte-wise keys, value
round trips, identity/widening limits, and lane-split columns on host and GPU (`Fp8ArgKeyDevice.cu`).
`thor_fp8_arg_key_microbenchmark` (EXCLUDE_FROM_ALL, not in the library) times the ARG update against a load-only
kernel with the same loads and output traffic, and is the small kernel to inspect in SASS. 1104/1105 GPU results: the
update is ~8 SASS instructions per element and matches the load roofline with 4 or 8 components per lane (geomean
1.03/1.04); 16 components per lane is schedule-sensitive and loses where warps are scarce, and capping it at 48
registers made it worse. ARG lanes therefore own at most 8 components; an ARG twin of a 16-component VALUE unit uses
twice the lanes under the same selector identity.

Planner (ARG-P1, 1106): original FP8 dense ARG is stamped from `planDenseReduction()`, like VALUE. Today every planned
ARG pass is a direct pass on the existing ARG kernels; the FP8 ARG families will be proposed from
`selectDensePhysicalPass()` for original-input FP8 RK sites, and their staged first stages produce
`[outer, shards, inner]` FP32 value + carried-index intermediates the planner already models.

OneWindow ARG twins (ARG-A2, 1107): `fp8OneWindowArgCompleteKernel` in `CubReductionNarrowFlatRCooperative.cu`, one
kernel per OneWindow unit (15; argmin/argmax x E4M3/E5M2 are the runtime `Fp8ArgFold`), with the VALUE unit's packet
windows, staging, ownership, R workers and merge topology and a `uint32` candidate per owned component. Candidates hold
the absolute row (R < 2^23 is part of `fp8OneWindowArgSupports()`), so lanes, workers and warps merge with an unsigned
min. The value is rebuilt from the key, the index is `row * domain_stride`. Selection
(`selectDensePhysicalPass`, 1142): an original FP8 ARG RK site ranks and autotunes its own candidates
(`Fp8RK::selectCachedArgRKSite`): the ARG twins of the complete retained VALUE MIN/MAX inventory plus the ARG-only
rotated passes (`retainedFp8ArgCandidates`), ranked by the shared geometry model with the ARG aggregate feature and
timed at the reduction's tuning depth as production runs them. The VALUE proxy and the 1128/1130 interim ARG-only rules
were removed; a site with no legal retained ARG pass uses the direct ARG reducer. The ARG kernels live in
the narrow TU's anonymous namespace as templates and are instantiated only by `launchFp8OneWindowArgComplete`, so the
isolated VALUE inventory (90 OneWindow kernels) is unchanged; ARG SASS is not yet in `fp8_selected_inventory_size.py`.
Open: the O1 c16 twin keeps 16 components per lane (the K-split needed for <= 8 changes the window staging); measure
it before deciding.
1107 GPU: 15 twins at 38-48 registers (O1 c16: 63, under the 65 cliff); same-unit ARG/VALUE 1.16x time, 1.08x after
normalizing for the 8-byte ARG output.

DeepRSmallK ARG twins (ARG-A3, 1108): `fp8DeepRArgFirstStageKernel<components 1/2/4/5/8>` and
`fp8DeepRArgTinyKFirstStageKernel<8/4 lanes>` (7 kernels, the VALUE ownership by K), the VALUE staged first stage's grid,
row groups and load-ahead rule with `uint32` candidates and four rows of a component per key word. They write FP32 value
+ carried index partials `[outer, shards, K]` (index = row * stride); the planner reduces the shard run with the
carried-index ARG reducer (stride 0 for the intermediate run). The depth is runtime for every ownership (VALUE compiles
it in for one component per lane). `fp8DeepRSmallKArgSupports()` = `Fp8DeepRSmallKPolicy::legal`.
1108 GPU: 7 kernels at 34-40 registers; same-depth ARG/VALUE 1.27x (tiny 1.23-1.27, c1 1.24, c2 1.16, c4 1.13, c5 1.19,
c8 1.92). 1109: four or more components per lane pack four components of one row per key word (at K 161..256 a group
holds one to three rows, so row packing left most of each key word empty); one and two components keep row packing.

RuntimeK ARG twins (ARG-A3, 1110): `fp8RotatedArgKernel<8|16>` in the narrow TU, the rotated sharded layout of
`awkwardAlignmentRotatedShardedFirstStageKernel` (16 cooperative warps, warp-aligned row windows that stay aligned every
16 rows; grid x = shard, y = K tile, z = outer) with `uint32` candidates and a runtime shard depth. Its K tile is the
packet span less the alignment reserve (249 for 8-byte packets, 497 for 16), so one kernel serves both VALUE tile
layouts. A rotated VALUE choice keeps its depth (staged, FP32 value + carried index partials); PacketComplete becomes the
single-shard Complete pass. Production uses 8-byte packets (`FP8_RUNTIME_K_ARG_PACKET_BYTES`, the <= 8 components per
lane rule); the sweep times both widths (`runtime_k_arg` p8/p16, staged depths and Complete) to confirm or flip it.
1110 GPU: rotated p8 39 registers, p16 53; best p16 / best p8 1.56x over 2408 cells (p16 faster on 78), so 1112 compiles
only p8. Staged RuntimeK production cells 1.13x VALUE; the PacketComplete stand-in (rotated, one shard) was 2.2x VALUE
(up to 3.5x at R = 4..16, K 257..513, many outputs: 16 cooperative warps over a handful of rows).

RuntimeK PacketComplete ARG twin (1112): `fp8PacketCompleteArgKernel<8>`, the topology of
`fp8GeneralRParallelKTileP16CompleteKernel` (persistent 256-thread blocks, one work item per [outer, K tile], VALUE's
1/2/4/8 warps per item over R, a staged per-row tile window with load-ahead) with `uint32` candidates. Lanes own at most
8 components, so VALUE tiles wider than 248 are halved (`fp8PacketCompleteArgTileComponents`) and the window uses 8-byte
packets (prefix <= 7). Four components of one row share a key word. Complete; names `packet_complete_wW_tileT`.
1127 replaces the window and the tile rule (ARG-A3j below): `fp8PacketCompleteArgKernel<16>`, tiles up to 256.

AlignedK ARG twins (ARG-A3, 1111): `fp8AlignedKArgKernel<2|8>` in the narrow TU, the K-parallel ownership of the VALUE
AlignedK kernels (one thread owns one aligned 2- or 8-byte packet of components and walks its shard's rows; no
cross-thread reduction; grid x = shard, y = K tile, z = outer; VALUE's balanced shard rows and persistent-grid rule for
tiny Complete items). The CTA size (32/64/128) is a launch operand, so two kernels serve VALUE's six packet/CTA pairs.
Eight-byte packets key two words per row (8 components per thread); two-byte packets pack two rows per key word. The
twin keeps VALUE's packet, CTA and shard count: one shard is Complete (any value storage, UINT32/UINT64 index), several
write FP32 value + carried index partials (index = (shard row base + row) * stride). `fp8AlignedKArgSupports()` adds the
23-bit row field to VALUE's legality (K a multiple of the packet; packet-aligned base). The sweep times every VALUE
variant (`aligned_k_arg`, named like `aligned_k`: `p{P}_cta{T}_shards{S}`).
With this, every retained VALUE family has an ARG twin.
1111 GPU: 289 tests pass; P2 28 registers, P8 38; no direct ARG passes left (ARG/VALUE 1.355 -> 1.227). Same-name
twin/VALUE 1.52 (best-vs-best 1.42), worst at R = 4 Complete (4.4x): the twin stored one scalar per component with lanes
8 elements apart and keyed rows without load-ahead. 1113 takes VALUE's R loop (batches of 4 P8 / 8 P2 rows loaded one
batch ahead) and writeback (P8 through a per-warp shared transpose into 16-byte lane stores, P2 8-byte lane stores;
`storeFp8ArgResults<N>`, element stores for other storage or misalignment).

## Cache-miss measurement (not a production executor)

1171: RK selections live in the one dense reduction selection cache
(`../DenseReductionSelectionCache.h`, kind `Fp8RK`; R / KR selections are kind `Fp8KR`), a bounded, same-key
single-flight cache of immutable recipes that also applies the depth rule (a deeper selection, completed or in
flight, answers a shallower request). The RK entry is the `Fp8RKSelection` physical recipe only, keyed by the
`Fp8RKSelectionKey` descriptor (`Fp8RKSelectionKey.h`, `toDenseReductionSelectionKey`). It owns neither Tensors,
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
    thor_dense_retained_fp8_rk_execution_smoke -j

./thor_fp8_rk_isolated_site_smoke
./thor_fp8_rk_selected_continuation_smoke
./thor_fp8_rk_selected_continuation_gpu_smoke
./thor_fp8_rk_stamp_autotuner_smoke
./thor_fp8_rk_stamp_lifecycle_smoke
./thor_fp8_rk_production_cutover_smoke
./thor_dense_retained_fp8_rk_execution_smoke
```

The cache's own rules (LRU bound, single flight, failure/retry, clear racing a flight, depth, kinds, recursion) are
`DenseReductionSelectionCache.*` in `thor_tests`.

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

### Packed staged ARG partials (ARG-A3g, 1116)

1113 traffic analysis: after normalizing for the 8-byte value + index output, large-R ARG was ~1.15x VALUE, and
staged families also moved twice VALUE's partial bytes (FP32 value + UINT32 index per partial, written and read back by
the generic carried-index reducer). 1114 ruled out per-element ALU (the eager p8 key update stays at 1.00-1.07x the load
roofline even at 16 warps/SM; a lazy byte-SIMD variant was no faster and used more registers).
Staged twins (DeepRSmallK, rotated RuntimeK, staged AlignedK) now write one 4-byte lane candidate per partial
(`key << 24 | shard row << 1 | negative zero`, shard-relative; `Fp8ArgOutputs::packed`) into the pass's own scratch, and
`fp8ArgPackedContinuationKernel` (one thread per output component, coalesced 4-byte reads per shard) widens each with its
shard's first row (uniform depth, or VALUE's balanced split for AlignedK), takes the minimum and writes the requested
outputs. To the planner a staged twin is one Complete pass with scratch `outer * shards * K * 4` bytes
(`retainedFp8ArgScratchBytes`); pass names end in `+fp8_arg_packed_continuation`.
1116 GPU: 294 tests pass. value+index ARG/VALUE 1.196 -> 1.161 (cells 0.972x; 388 >5% faster, 46 slower); index-only
1.126 -> 1.095 (664 of 1600 cells <= 1.05; staged families 1.05-1.06). DeepRSmallK same-depth twin/VALUE 1.17 -> 1.05.
The wider `Fp8ArgOutputs` cost the Complete kernels (OneWindow O1 c16 63 -> 64 registers, 1.12x slower; PacketComplete
1.04x); 1117 compiles the `packed` branch only into the staged kernels (`storeFp8ArgResult<true>`).

### A5 (1118): inventory and fallback

- `benchmarks/fp8_selected_inventory_size.py` counts the ARG twins as family `ARG`: 27 kernels (15 OneWindow units,
  7 DeepRSmallK ownerships, rotated RuntimeK P8, PacketComplete P8, AlignedK P2/P8, packed continuation). Argmin/argmax
  and E4M3/E5M2 are the runtime fold, so there is no dtype or operation axis.
- `retainedFp8ArgRotatedDeepPass` (was `retainedFp8ArgFallbackTwin`; 1172 removed its `fallback` flag): the rotated
  RuntimeK ARG pass (depth min(R, 1024), packed partials when staged) is in every ARG candidate set, so a site whose
  VALUE identities have no legal twin is still covered instead of reaching the generic direct ARG reducer. Original FP8 RK ARG reaches the direct FP8 tiled ARG kernels only beyond the rotated kernel's own limits
  (2^31 - 1 grid extents), so their FP8 instantiations are the next removal candidate once a sweep shows no `direct:`
  FP8 RK pass.
- 1118 GPU: 295 tests pass; inventory ARG 27 kernels, 523,008 SASS bytes (5.9% of the 8,798,336-byte selected
  first-stage total); value+index sweep unchanged (1.162) with no `direct:` or `_fallback` FP8 RK ARG pass.
- 1119: `launchArgReductionForInput` / `launchComposedArgReductionStageForInput` no longer compile the tiled direct ARG
  kernels for one-byte (FP8) input; that case throws (unreachable: original FP8 RK ARG always plans a retained FP8 ARG
  pass, and later passes consume FP32). FP8 R/KR sites (contiguous, transform-reduce) keep their direct paths.

### VALUE-P1 (1120, A/B): packed FP8x4 decode in DeepRSmallK VALUE

`decodeNarrowFp8DeepRGroup` replaces the first stage's scalar byte-at-a-time decode: four or more components per lane
decode four components of one row per `decodeFp8x4ToFp32`; one or two components per lane decode four consecutive
rows of one component (the ARG twin's 1109 layout, which ran 0.93x VALUE at eight components per lane). Row order per
component is unchanged, so results are bit-identical (host-checked against the scalar loop for SUM/PRODUCT/MIN with the
square transform). `THOR_FP8_DEEP_R_VALUE_SCALAR_DECODE=1` restores the scalar loop for the A/B.
1120 GPU A/B (MIN/MAX, `--focus=arg`, same-name VALUE deep_r_small_k units vs 1119; controls OneWindow 0.999, ARG
units 1.000): c8 0.828 (rows64 0.90 ... rows1024 0.69), c5 0.960, c4 0.985, c2 1.010, c1 1.034 (rows64 1.06), tiny-K
1.00. Registers: c8 48, others 39-40 (under the 49 / 41 cliffs). DeepRSmallK selected SASS 2,053,120 -> 1,971,456 bytes
(-4.0%). 1122 keeps the packed decode for four or more components per lane only; one and two go back to the scalar loop.
1122 SUM/PRODUCT A/B: c8 0.870, c5 0.979, c4 0.994; c1/c2/tiny 1.000 (control). Kept for every operation.

### ARG-A3h (1124): reduce four rows inside the key word

With one or two components per lane, OneWindow ARG loads four consecutive rows of one component per word, and its
inner loop built four lane candidates per word (per-byte shift, mask, row arithmetic and min: 82 SASS instructions per
four rows against VALUE's 54 at O1 c1). `fp8ArgRowsCandidate4` (Fp8ArgKey.h) instead spreads each byte's key, its
negative-zero bit and its row slot 0..3 into one 32-bit word with four `__byte_perm`s, takes the unsigned minimum of the
four words (key first, then the lower slot, then -0) and only then builds the single lane candidate. Ties go to the
lowest row as before, so results are identical (host-checked against the per-byte minimum, `RowsCandidate4MatchesBytewise`).
Used by the OneWindow c1/c2 four-row main loop and by DeepRSmallK `deepRArgRows4` when all four rows are present; tails
keep the per-byte path.
1124 GPU (index-only, ARG unit time vs 1117): OneWindow O1 c2 load-ahead 0.959-0.967 (K 33-64: 0.91-0.97), O1 c1
0.990-0.994, O2 c1 0.994, O8 c1 0.999; DeepRSmallK ARG 0.992-0.999; registers unchanged. But O2/O4/O8 c2 (no
load-ahead) 1.035/1.022/1.018, up to 1.17x at large R. 1125 keeps the four-row candidate only where it measured
faster (one component per lane, or load-ahead) via `keyOneWindowArgGroup<..., RowsCandidate>`; the other c2 units
compile the 1122 per-byte loop again.
1125 GPU: O2/O4/O8 c2 back to 0.999-1.000 of 1117; O1 c2 load-ahead 0.96, O1 c1 0.99 kept. Accepted.

### ARG-A3i (1126): linear staging for OneWindow O1 c1/c2 ARG

The O1 c1 worst cells (K 17-32, ARG/VALUE up to 1.44) are ALU-bound: only K of 32 lanes key, and each four-row word
spent ~41 of its SASS instructions gathering four bytes from the plane-major staging (per byte: offset, mask, plane
index, word load, extract). With one subgroup per warp (O1) nothing needs the plane-major layout, so the O1 c1/c2 ARG
units stage each lane's 16-byte packet with one vector store at byte lane * 16 and read staged byte b as one LDS.U8;
four rows merge with three `__byte_perm`s. Lanes read at most 32 contiguous bytes per row step, so reads stay free of
bank conflicts. O2/O4/O8 (several subgroups whose linear slices would share banks) and the c4+ buckets keep
plane-major staging; the VALUE kernels are unchanged. Results are identical (same bytes, same order of candidates).
1126 GPU (index-only vs 1125, ARG unit time): O1 c1 0.888-0.900 (p10 0.80), O1 c2 load-ahead 0.846-0.860 (p10 0.73),
every other unit 0.998-1.001; ARG/VALUE same-name O1 c1 1.115 -> 1.002, O1 c2_la 1.159 -> 0.979; c2_la 48 -> 44
registers; ARG SASS 524,032 -> 520,960 bytes. Overall index-only ARG/VALUE 1.091 -> 1.065 (925 of 1600 cells <= 1.05).

### ARG-A3j (1127): PacketComplete P16 window, quad ownership, fewest-word tiles

In 1126, 26 production PacketComplete cells (index-only ARG/VALUE 1.13-1.42) were nearly all VALUE tiles of 249-410
components, which the 248-component ARG kernel halved: 257 -> 129 needs five slots, two key words per row for barely
one word of data (K = 513: 7 words per row against a minimum of 5). The kernel also gathered every byte from the
plane-major staging. 1127: 16-byte packets (512-byte window, prefix <= 15) allow 256 components per tile with the same 8
candidates per lane; the window is staged linearly (one 16-byte store per loading lane) and lane L owns components
q * 128 + 4L .. +3 for quads q = 0, 1, so each key word is two LDS.32 and one funnel shift (the prefix's byte offset is
warp-uniform). `fp8PacketCompleteArgTileComponents` keeps VALUE's tile (capped at 256 and K) unless a 256-component
tile needs fewer key words per row (sum over tiles of ceil(tile / 128); 256 always reaches ceil(K / 128)). Host-checked
against the per-byte reference (1.05 M components, random K, tiles, prefixes); results are identical. The input base
must be 16-byte aligned, as for the VALUE kernel.
1127 GPU (index-only vs 1126, production PacketComplete cells): 0.79-0.99 on 21 of 26 cells (K = 513 R = 16 1.41 ->
1.13x VALUE, K = 510 R = 8 1.43 -> 1.14x); two 24 MiB R = 4 cells slower (1.16, 1.20; small-cell noise is up to ~12%).
PacketComplete 55 registers. ARG SASS 520,960 -> 505,856 bytes. Overall 1.065 -> 1.062.

### 1128: OneWindow padded linear staging (all c1/c2), O1 c4/c8 quads, few-item PacketComplete -> rotated

- Every OneWindow ARG unit with one or two components per lane now stages linearly (1126 did O1 only). O2/O4/O8
  slices are padded by LanesPerOutput words so consecutive subgroups start 16/8/4 banks apart and their reads (at
  most LanesPerOutput contiguous bytes per row step) never share a bank. Remaining 1127 gaps: O2 c1 1.11, O2 c2 1.17,
  O4 c2 1.13, O8 c1/c2 1.07.
- O1 c4 and c8 (1.14 / 1.10) stage linearly and own quads of adjacent components (components q * 128 + 4L .. +3),
  so a key word is two LDS.32 and a funnel shift, as in PacketComplete. O1 slices carry a 4-word pad for the second
  word. O1 c16 (63 registers, next to the 64 cliff) keeps the plane-major lane + 32 * slot ownership.
- A PacketComplete VALUE choice with fewer than 8192 work items (outer x ARG K tiles) and 16 <= R <= 1024 runs the
  single-shard rotated RuntimeK ARG pass (`p8_complete`): in 1127 it measured 0.88-0.96x VALUE on the 1.7-7k-item
  cells where PacketComplete was 1.15-1.21x, while PacketComplete won from ~9k items up.
1128 GPU (index-only vs 1127): O2 c1 0.892, O2 c2 0.840, O4 c2 0.879, O8 c1/c2 0.93, O1 c4 0.89-0.92, O1 c8
load-ahead 0.88-0.92 (48 -> 56 registers; W4 R <= 12 cells 1.06-1.21, combinations VALUE does not select), rotated
routing 0.77-0.91 on its four cells. Same-name OneWindow ARG/VALUE 1.047 -> 1.008; overall 1.062 -> 1.039 (1148 of
1600 cells <= 1.05).

### 1129: DeepRSmallK one-component workers at K 9-16; quad ownership for OneWindow O2/O4/O8 c4

- DeepRSmallK production cells by K (1128): K 9-16 1.24 (worst 1.41), every other bucket 1.03-1.08. At K 9-16 the
  one-component ownership keyed with only K of 32 lanes. floor(32 / K) workers (3 at K 9-10, 2 at K 11-16) now share
  a component, worker w keying rows w, w + workers, ... of each staged group (the tiny-K loop with a runtime lane
  count), folded with shuffles before the cross-warp partials. K 17-32 (one worker) runs the former loop.
- OneWindow O2/O4/O8 c4 (1.000 / 1.088 / 1.048) take the O1 c4 quad ownership on their padded linear slices. O2/O4
  c8 (47 registers) and O1 c16 (63) keep lane + slot ownership.
1129 GPU (index-only vs 1128): DeepRSmallK production K 9-16 1.24 -> 1.05 (worst 1.41 -> 1.23); O2/O4/O8 c4
0.883/0.843/0.891 (same-name ARG/VALUE ~1.00); c4 units 39 -> 40 registers, DeepR c1 36 -> 40. Overall 1.039 -> 1.033
(1181 of 1600 cells <= 1.05; production OneWindow 1.006).

### 1130: ARG-specific choices for DeepRSmallK depth and shallow PacketComplete

- DeepRSmallK twins deepen shards below 256 rows to 256 (when R > 256). In 1129 the ARG depth that mirrors VALUE
  cost 1.062 over the 337 production cells; a 256-row floor 1.053 (the per-cell best depth would be 1.037).
- A PacketComplete VALUE choice with R <= 12 runs the OneWindow O1 W1 ARG unit when one covers K (1129: 0.99-1.17x
  VALUE against PacketComplete's 1.08-1.37x at K = 257); otherwise, with R <= 8 and an even K, AlignedK P2 CTA32
  in one shard (1.02-1.28x against 1.09-1.36x at K = 510); otherwise PacketComplete (K = 513: nothing faster).
1130 GPU (index-only vs 1129): shallow K = 257 PacketComplete -> OneWindow 0.73-0.99 (ARG/VALUE 1.00-1.18); K = 510 ->
AlignedK P2 0.93-1.07 (a wash); DeepRSmallK production 1.062 -> 1.057; overall 1.033 -> 1.032. ARG work closed here.

### 1131: launch limits in selection (UINT64 survey gaps 1, 2, 4)

The 2026-11-01 survey (project doc `surveys/Thor_FP8_UINT64_Survey_2026-11-01.md`) found that no retained VALUE
selection policy encoded the 32-bit limits its launcher throws on, and the autotuner does not catch launcher
`logic_error`s, so an FP8 input of 2^32 or more elements could rank OneWindow or PacketComplete and fail the whole
stamp (ARG mirrors VALUE and failed with it). `Fp8RKLaunchLimits.h` restates the limits and each policy applies them:
OneWindow `select()` / `shortlist()` require outer * R * K <= 2^32 - 1 (32-bit byte offsets); DeepRSmallK, AlignedK
and rotated RuntimeK require reductionWorkGrid's extents (shards, tiles, outer <= 2^31 - 1; a static_assert ties the
constant to REDUCTION_WORK_GRID_MAX_EXTENT); AlignedK requires rows per shard <= 2^32 - 1; PacketComplete requires
outer, K <= 2^32 - 1 and R <= 2^32 - 9. The PacketComplete VALUE launcher no longer requires outer * R * K <= 2^32 - 1:
its kernel's input and output offsets are 64-bit, so it now runs inputs of 4 GiB and more (gap 2). The DeepRSmallK ARG
twin uses the DeepRSmallK legality and inherits the grid caps (gap 4). Shapes that launched before are unaffected; only
candidates that would have thrown are removed. Gap 3 (FP8 ARG with R >= 2^32: the packed continuation's 32-bit wide
index) is open.
1131 GPU: tests pass (`Fp8RKLaunchLimits.*` included).

### 1132: 55-bit widened ARG index (UINT64 survey gap 3)

Every staged FP8 ARG plan ends in `fp8ArgPackedContinuationKernel`, which widened shard-relative lane candidates to
`(key << 33) | (index << 1) | nz` with a 32-bit row base: R had to stay <= 2^32 - 1, the planner did not check it, and
larger R threw at run time. The widened candidate is now `(key << 56) | (index << 1) | nz` with a 64-bit row base
(`fp8ArgWiden(uint32_t, uint64_t)`, `fp8ArgWideIndex` -> uint64), so indices up to 2^55 - 1; order (key, then index,
then -0) and the identity are unchanged. The continuation's limits are now R - 1 <= 2^55 - 1 and shards <= 2^31 - 1.
First-stage and Complete kernels are untouched (their 32-bit lane candidates hold shard-local rows < 2^23), so only
the continuation's arithmetic changed (64-bit row base, shift and mask constants); its traffic is the same.
`ContinuationIndexesPastFourBillionRows` runs the continuation on synthetic partials of three 2^31-row shards.
