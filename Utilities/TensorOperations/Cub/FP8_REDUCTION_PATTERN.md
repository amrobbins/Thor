# The FP8 dense reduction pattern (template for FP16, BF16 and FP32)

Status: 1172, written when FP8 reached the end of its plan (tests passing, rankers trained, framework integration
done). 1231: FP32 follows the pattern too (section 8). 1262: FP16 / BF16 follow it (section 9), so every dense
reduction dtype now runs the pattern and the legacy dense families are deleted (1258-1261). This
document describes the pattern FP8 settled on, the rules it keeps, and what another dtype needs to follow it. The
history and measurements behind each choice are in `Fp8RK/README.md`, `../../../benchmarks/*.md` and the project's FP8
completion plan. `ReductionArchitecture.md` is the source-ownership map.

## 1. What the pattern delivers

A dense reduction (`CubReduction`, `CubArgReduction`) of an FP8 E4M3 or E5M2 tensor over any set of axes:

- runs as a short sequence of physical passes, each a Thor-owned kernel chosen for that pass's geometry on the
  running GPU;
- reaches line rate where the hardware allows it. The target is about 1.5 TB/s at large sizes, against a 1.59 TB/s
  copy on the development GPU. Most large shapes reach it (KR VALUE at 64 MiB and above: every cell; ARG: 97–98%);
  the worst large KR cells sit around 1.3 TB/s;
- costs a bounded, predictable stamp time (median 11–35 ms per tuned site, by size) and nothing at run time (no
  planning, tuning or allocation in `run()`);
- is deterministic in its results: no atomics, fixed combining order, so a plan gives the same output bits on
  every run. The plan itself is cached per process. Between processes, a site whose best shapes are within
  measurement noise of each other may pick either of them: each is legal, and they are equally fast;
- covers every public operation (Sum, Mean, Product, Min, Max, L1Norm, L2Norm, SumSquares; argmin and argmax with
  value, index or both, UINT32 or UINT64 indices).

## 2. The layers

```text
CubReduction(op, axes, dtype, scale, depth?)               depth: explicit, else the network's / ambient depth
   │ stamp()
   ▼
canonical problem (DenseReductionPlanning)                 runs: kept / reduced, adjacent runs merged
   │
   ▼
planner (DenseReductionPlanner) ── one pass at a time ──►  site = one reduced run; family R, KR or RK
   │   lazy best-first: visit sites by the bytes left if   (KR before RK, outer before inner on ties)
   │   the site were fully reduced; stop when no unvisited
   │   bound can beat the best candidate
   ▼
physical selector (DenseReductionPhysicalSelector)         routes original-FP8 sites to the retained selectors
   │
   ├─ RK site ──► Fp8RK::selectCached*RKSite ─┐
   └─ R/KR site ► Fp8KR::selectCached*Site ───┤
                                              ▼
                          DenseReductionSelectionCache     hit: the recipe; miss: one flight per key
                                              │ miss
                                              ▼
                 candidate inventory ──► ranker (top N by depth) ──► contest (ReductionAutotuning::autotune)
                 every legal, resident       compiled XGBoost trees     cold-L2 timing on this GPU,
                 kernel shape                over C++ features          shared probe/batching policy
                                              │
                                              ▼
                      winner = immutable physical recipe (cached), then the next problem from its output
```

Each layer and the rules it keeps:

### 2.1 Canonical problem and planner

- `DenseReductionPlanning` turns (dimensions, axes) into runs of kept and reduced extents, merging adjacent runs of
  the same kind. A site is one reduced run. Its family follows from position: **R** (the whole tensor), **KR**
  (reduce the innermost run: contiguous rows), **RK** (reduce a middle run: strided rows `[outer, R, K]`).
- A pass may reduce its site completely or partially. A partial pass leaves an intermediate run, which becomes a new
  site of the next problem. Staged kernels are therefore ordinary candidates: the planner does not special-case
  them.
- 1167 site order: for each step, the bound of a site is the bytes the next tensor would have if that site were
  reduced completely. Sites are visited in ascending bound (ties: KR before RK, then the outer run first). The planner
  keeps a complete pass on the best bound, and tries further sites only while their bound can beat the best candidate
  so far. This gives the same plans as selecting every site, with fewer contests (`CubDenseReductionLazyPlannerTest`).
- Original-input transforms (abs, square) are applied once, on the first pass. Every non-final output is FP32 (ARG:
  FP32 value plus carried index). Finalization (Mean's divisor, L2Norm's square root), the runtime output scale and
  the conversion to the output dtype happen only on the last pass.

### 2.2 Kernel inventory (per dtype)

The retained kernels are Thor-owned CUDA, templated on the FP8 format, the input transform and the combine operation.
Each family covers a region of geometry:

| Family | Sites | Shapes |
| --- | --- | --- |
| OneWindow | RK | runtime-K bucketed; O outputs per CTA, W warps, P packets; complete |
| DeepRSmallK | RK | deep R, small K; staged by depth (FP32 partials `[outer, shards, K]`) |
| AlignedK | RK | K a multiple of the packet; complete or staged by shard count |
| RuntimeK | RK | runtime layout (PacketComplete, rotated prefix); complete or staged |
| Pack (and vector Pack) | R / KR | short rows (≤ 32 B): one thread owns 2–16 rows |
| Group | R / KR | G lanes (1–32) per row |
| Block | R / KR | one CTA per whole row (rows ≥ 1 KiB) |
| Split | R / KR | chunks of long rows with FP32 partials; a finishing kernel combines them in chunk order |

ARG has its own kernels: an ARG twin of each VALUE MIN/MAX RK shape where one is legal, the rotated RuntimeK ARG pass
(in every candidate set, so every site is covered), packed staged partials with an in-pass continuation, and the KR ARG
segment kernels. FP8 ARG compares keys built so that one integer comparison orders values, NaN and ±0 as the public
semantics require (`Fp8ArgKey.h`). MIN/MAX VALUE fold into one MIN domain.

Every family has a host-side policy (`Fp8RK/Fp8*Selector.h`, `Fp8KR/Fp8KRSegments.*`) that answers three questions
without CUDA: is a shape legal for this geometry, how much scratch does it need, and which shapes exist (the pool). The
CUDA side answers residency (CTAs per SM of the exact compiled specialization). Only legal, resident shapes within the
workspace budget become candidates.

### 2.3 Ranker

- One model per site family: RK (`Fp8RK/Fp8GeometryRankerModel.h`, 26 features) and KR (`Fp8KR/Fp8KRRankerModel.h`,
  24 features). Each covers VALUE and ARG, both formats and every operation, with the semantics as features.
- Features are computed in C++ (`rankerFeatures`, `fp8KRRankerFeatures`). The training sweeps write the same C++
  features beside each timing, so training never re-implements them in Python.
- Training: XGBoost `rank:ndcg` grouped by cell; labels grade regret against the cell's fastest candidate (1%, 3%,
  7%, 15%); folds never split a shape; held-out shapes are only for validation.
- Export: the trees become a generated C++ header, evaluated in float32 in a fixed order. The exporter checks the
  C++-order evaluation against XGBoost's margins, and writes parity fixtures; `ModelMatchesTrainingFixturesExactly`
  checks the installed header against them bit for bit.
- The ranker never decides alone: it orders the inventory, and the contest times the top N. A weak ranker costs
  speed (the best shape falls outside the top N), never correctness.
- Retraining for a GPU: `benchmarks/FP8_RANKER_RETRAINING.md`.

### 2.4 Contest (one policy for every reduction)

`ReductionAutotuning::autotune` (`ReductionAutotuner.h`, `ReductionAutotunerCore.h`) times the candidates on a
representative input. The policy lives in `ReductionAutotuning::Options` defaults; selectors set only rounds and trim
from the depth:

- **Cold L2.** Each launch reads its own slot of a ring sized for a reuse distance of 8 × L2 (slot = input + output +
  scratch, 256-byte aligned), so every timed launch starts L2-cold. Nothing is flushed between launches, and no
  allocation or copy is inside a timed interval.
- **Probe, then batch.** After one warmup, every candidate gets one timed probe launch. Each candidate's launches per
  measurement are then clamp(ceil(200 µs / probe), 1, 32): CUDA events resolve about 2 µs, so short kernels are
  batched until the timer is about 1% of the interval, and slow ones are not repeated.
- **Abandon the hopeless.** A candidate whose probe is over 3 × max(best probe, 8 µs) is not measured further.
- **Rounds and trim** (`fp8RKTuningPlan`): Quick 5 rounds (trim 1), Regular 7 (trim 1), Ultra 11 (trim 2), interleaved
  across candidates in a random order. The fastest and slowest `trim` rounds of each candidate are dropped, and the
  smallest trimmed mean wins (ties: the higher-ranked candidate).
- **One measurement at a time per GPU.** `ScopedReductionGpuMeasurement` serializes contests, and the allocation
  and release of the representative input, on each device. Concurrent stamps of different sites queue there.

Evidence for this policy: with fixed launch counts, a stamp's wall time was 4 ms plus its contest's GPU time, and below
16 MiB 92–98% of it went to shapes 10–100× slower than the winner. With probes, batching and abandonment, the median
stamp fell to 11–35 ms with picks as good (1161 KR, 1164–1165 RK).

### 2.5 Depth and cache

- `CubReductionTuningDepth` (Quick, Regular, Ultra) sets N and the rounds: RK 6 / 12 / 24, KR 6 / 12 / 12. A
  reduction's depth is explicit, else the network's (`Network::setReductionTuningDepth`, Regular when unset, applied
  by `Network::place()`), else the calling thread's scope or the process default.
- `DenseReductionSelectionCache` is the one cache of every selector: a bounded LRU of immutable recipes keyed by
  (selector kind, device identity, depth, site words). It has single flight per key, does not cache failures, and
  `clear()` cannot be undone by a running flight. A deeper entry, completed or in flight, answers a shallower request.
  A recipe holds no Tensor, Stream, executable or workspace (`AcceleratorBackendSelectionRecipe`). The cache is not
  persisted on disk (user decision, 1169).
- A cache hit costs no CUDA query and no allocation (device identity is cached per ordinal).

### 2.6 Stamped execution

Stamping allocates each non-final intermediate and each pass's scratch, and records the physical passes. `run()`
launches them. Workspace today is the sum over passes. Two ping-pong buffers would be enough, but that change waits
until CUB is out of the reducer (user decision, 1168).

## 3. Rules the pattern keeps

These are constraints, not preferences. A new dtype keeps all of them.

1. **sm_80 and later.** No sm_90-only mechanism (no PDL, no TMA, no clusters) in the production path.
2. **No atomics** in reduction kernels. Multi-CTA work writes partials and finishes in a fixed order (Split's
   finishing kernel, staged passes). Results are deterministic run to run.
3. **Every candidate is legal and compiled.** Selection never guesses a layout or occupancy. It asks the policy for
   legality and the compiled kernel for residency, and fails closed.
4. **The contest measures; the model only orders.** No model margin is ever read as microseconds.
5. **One timing method.** The contest and the training sweeps measure the same way (cold inputs, cold outputs, probes,
   ~1 ms or ~200 µs batches). Changing the method means re-sweeping, or the ranker learns a different machine.
6. **No tuning, planning or allocation at run time.**
7. **Unsupported means an error, not a silent fallback.** Where the retained kernels cannot run a case, the reducer
   throws rather than routing to CUB or a generic kernel unannounced. FP8 meets this everywhere since 1173: every
   original FP8 R, KR and RK site, VALUE and ARG, is planned by a retained selector or rejected
   (`CubDenseFp8NoFallbackTest`). Sites over FP32 intermediates use the FP32 path, which meets the rule too since
   1225 / 1227 (section 8).
8. **Report bandwidth in TB/s, measured reproducibly** (moved bytes over time). TB/s is the measure (user, 1177). A
   comparison with a device copy of the same bytes is a secondary view, used where it helps (cells near the launch
   floor).
9. **CUDA builds with `--Werror all-warnings`.**

## 4. How it was measured, and what went wrong first

Each of these was a real mistake in the FP8 work. The current method exists to avoid them.

- **One launch per sample.** CUDA events resolve about 2 µs, so a 5–10 µs kernel timed once sits on a 20–40% grid
  (1162/1163 sample spreads of 8–22% below 16 MiB). Batch launches into ~1 ms samples (sweeps) or ~200 µs
  measurements (contests).
- **Warm outputs.** A shared output smaller than L2 stays resident between launches and hides write traffic (moved
  TB/s read above the copy rate). Rotate output views too.
- **Fixed launch counts in contests.** These made small sites cost more to tune than large ones. The fix is probe,
  batch, abandon.
- **Comparing a production stamp to a sweep candidate timed differently.** This disagreed by up to 28%. Time both
  the same way before comparing them.
- **Blaming the wrong cause.** Ring allocation looked like the stamp cost, and alignment looked like a source of
  spread. Each was disproved by a regression or an A/B. Measure the hypothesis before fixing it.

## 5. Replicating the pattern for another dtype

What is shared already (do not copy it):

- The planner, problem representation, lazy site order and pass advancement.
- `ReductionAutotuning` (the contest policy), the GPU measurement lock and the representative-input lifecycle.
- `DenseReductionSelectionCache`: add a `DenseReductionSelectionKind`, not a cache.
- The tuning depth, the network setting and the ambient scope.
- The training method (`benchmarks/ColdTiming.h`) and the trainer, evaluator and exporter
  (`benchmarks/fp8_ranker/{evaluate,export}.py`, which the KR trainer already reuses).

Per dtype, in order:

1. **Audit first.** Time the current production reduction for the dtype (R, KR and RK, VALUE and ARG)
   in TB/s with `thor_dense_production_audit --dtype=…` and `benchmarks/dense_production_audit.py` (1174), and
   keep the run: it is the baseline every later step of the dtype is compared with (`--compare`). Every kernel family
   is carried over regardless (decisions below); the audit shows where the gaps are largest, which orders the work.
2. **Kernel inventory.** For each family, the kernels templated on the dtype, transform and combine, with the dtype's
   element width in the packet and ownership choices (2-byte FP16/BF16, 4-byte FP32). Staged passes write FP32
   partials; ARG writes its keys (a 16-bit dtype needs its own ordered-key encoding, FP32 has one). Keep the FP8
   invariants: transform once, finalize once, fixed-order finishing.
3. **Host policy.** Legality, scratch and the candidate pool for each family, CUDA-free and unit-tested like
   `Fp8RKLaunchLimitsTest` and `Fp8KRRanker` (`PoolStartsWithTheListAndIsLegalAndDistinct`).
4. **Selector.** A `selectCached…Site` per family that builds the key (new kind, site words that include everything
   that changes the shortlist, the measurement or legality), ranks, contests through `ReductionAutotuning::autotune`
   and returns a recipe deriving `AcceleratorBackendSelectionRecipeTag`. Route original-dtype sites to it in
   `DenseReductionPhysicalSelector`.
5. **Features and sweep.** C++ ranker features and a training mode in a sweep that writes them beside each timing,
   with the shared timing method.
6. **Train, export, install.** The same trainer flow, depth rules and parity fixtures; then the retraining guide's
   audits.
7. **Tests.** A randomized multi-step test like `CubReductionFp8MultiStepTest` (every operation, a random seed
   printed for reproduction, a few seconds per run), parity tests, and depth/cache tests for the new kind.
8. **No CUB for the dtype** from the start of its work (see the decisions below); any case the retained kernels
   cannot run is an error.

Decisions for every dtype (user, 2026-11-02):

- **One ranker per dtype** (per site family). Kernels for one dtype can then change, with that dtype's rankers
  retrained, without retraining any other dtype. FP8's E4M3 and E5M2 count as one dtype here: they share kernels and
  rankers.
- **Carry every kernel family over at first.** Prune a family for a dtype only on evidence from its sweeps. Wider
  elements mean less ALU work per input byte. Where ALU work was not the limit, every family may still be needed for
  2- and 4-byte dtypes.
- **Accumulate in FP32**, as today. Every reduction path accumulates in FP32: the CUB adapters `static_assert` it, and
  the FP8 kernels accumulate in `float` and write FP32 partials. FP64 is neither an input nor an output dtype
  (1258 / 1259).
- **CUB stops being a candidate as soon as work on a dtype starts.** It is not timed against the dtype's new kernels.
- **The FP8 families are the inventory for every dtype** (OneWindow, DeepRSmallK, AlignedK, RuntimeK; Pack, vector
  Pack, Group, Block, Split; their ARG forms). A dtype neither keeps other kernel families of its own (FP32's
  KParallel and RCooperative retire) nor adds new ones.
- **Shared template or copy, per kernel.** A kernel whose other-dtype form differs only by the input element type
  becomes a template shared across dtypes (the FP8 instantiations' SASS must stay unchanged). One whose shape logic
  differs is copied and lives independently for that dtype.
- **Port everything, then switch once** (FP32, 2026-11-02): a dtype's kernels, selectors and rankers are built and
  tested without routing production to them, then every site of the dtype switches in one patch.

## 6. Open items left by FP8

- **CUB fallback for FP8: closed in 1173.** The R/KR direct proposal, and the RK ARG generic proposal when no retained
  ARG pass is legal, are planning errors now.
- **FP8 depends on the FP32 path.** Staged FP8 RK VALUE passes write FP32 partials that the FP32 planner reduces, and
  the RK ranker was trained on times that include that FP32 continuation. When FP32 is overhauled, re-run the FP8
  audits of `benchmarks/FP8_RANKER_RETRAINING.md` (steps 1 and 6), and retrain the RK ranker if the picks or the
  top-N coverage got worse. Closed at 1230: after the FP32 switch the FP8 audit matches its 1176 baseline (per-cell
  ratio p50 1.000 everywhere; 2 of 600 cells below 0.95, at 0.943 and 0.949), so the FP8 RK ranker was not retrained.
- **Ping-pong workspace** (two scratch buffers instead of one per pass), after CUB is removed.
- **Representative tuning input.** Contests time on an input of 1.0 everywhere. The kernels' speed does not depend on
  the values for finite inputs, but a realistic input would also exercise the NaN and special-value paths of ARG.
- **Multi-GPU models.** One compiled model per ranker per build (see the retraining guide).
- **Pure R and ARG at 16–64 MiB** stay below 1.5 TB/s in some cells (fixed per-launch cost of Split's finishing
  kernel, occupancy-bound Group). KR kernel work stopped there by decision (2026-11-01).

## 7. Where things are

| Concern | Files |
| --- | --- |
| Problem, sites, advancement | `DenseReductionPlanning.{h,cpp}` |
| Site order and plan | `DenseReductionPlanner.{h,cpp}` |
| Routing to selectors | `DenseReductionPhysicalSelector.{h,cpp}` |
| RK inventory, policies, selection | `Fp8RK/` (see `Fp8RK/README.md`), `DenseRetainedFp8RKPhysicalPlan.*`, `DenseRetainedFp8ArgPhysicalPlan.*` |
| R / KR inventory, policies, selection | `Fp8KR/` (`Fp8KRSegments.*`, `Fp8KRStampAutotuner.*`, `Fp8KRRankerModel.h`) |
| Contest | `ReductionAutotuner.h`, `ReductionAutotunerCore.h`, `ReductionGpuMeasurementLock.*` |
| Cache | `DenseReductionSelectionCache.{h,cpp}`, `DenseReductionDeviceIdentity.cpp` |
| Depth | `CubReductionTuningDepth.{h,cpp}`, `Network::setReductionTuningDepth` |
| Sweeps and training | `benchmarks/Fp8RetainedColdL2Sweep.cpp`, `benchmarks/Fp8KRTrainingSweep.cpp`, `benchmarks/ColdTiming.h`, `benchmarks/fp8_ranker/`, `benchmarks/fp8_kr_ranker/` |
| Audits | `benchmarks/DenseProductionAudit.cpp` + `dense_production_audit.py` (any dtype), `benchmarks/fp8_rk_stamp.py`, `benchmarks/fp8_kr_training.py` |
| Retraining | `benchmarks/FP8_RANKER_RETRAINING.md` |
| Tests | `CubReductionFp8MultiStepTest`, `Fp8GeometryRankerTest`, `Fp8KRRankerTest`, `CubReductionTuningDepthTest`, `DenseReductionSelectionCacheTest`, `CubDenseReductionLazyPlannerTest`, `CubArgReductionFp8*Test`, `NetworkReductionTuningDepthTest` |
| FP32 (section 8) | `Fp32RK/`, `Fp32KR/`, `Fp32ArgKey.cuh`, `Fp32CarriedArgContestInput.*`, `DenseRetainedFp32ArgPhysicalPlan.*`; the FP32 kernels sit beside FP8's in the NarrowFlat, KParallel and AwkwardRotated TUs |
| FP32 tests | `CubReductionFp32RKTest`, `CubReductionFp32KRTest`, `CubReductionFp32RKSelectorTest`, `CubReductionFp32KRSelectorTest`, `Fp32RKCandidatesTest`, `Fp32KRRankerTest`, `CubReductionFp32RoutingTest`, `CubReductionElementWidthTest` |
| FP16 / BF16 (section 9) | `F16RK/` (`F16RKCandidates.*`, `F16RKArgCandidates.*`, `F16RKStampAutotuner.*`, `F16RKRankerModel.h`), `F16KR/` (`F16KRStampAutotuner.*`, `F16KRRankerModel.h`); the 2-byte kernels are the FP8 / FP32 kernels' typed instantiations in the same TUs |
| FP16 / BF16 tests | `CubReductionF16RKTest`, `CubReductionF16KRTest`, `CubReductionF16RKSelectorTest`, `CubReductionF16KRSelectorTest`, `F16RKCandidatesTest`, `F16KRRankerTest`, `CubReductionF16RoutingTest` |

## 8. FP32 (1174-1230)

FP32 was the second dtype through the pattern, and the first with work beyond its own sites: every staged pass of
every dtype finishes in FP32. The plan, its measurements and its decisions are in the project's FP32 plan
(`plans/Thor_FP32_Dense_Reduction_Plan_2026-11-02.md`).

### 8.1 What FP32 has

- **Every FP32-reading site is retained.** `selectDensePhysicalPass` sends original FP32 input and the FP32
  intermediates of every dtype's earlier passes, VALUE and ARG, to the FP32 selectors (`Fp32RK::selectCached*Site`,
  `Fp32KR::selectCached*Site`). Dense FP32 ARG runs the planned ARG executor. There is no fallback: a host-only
  context (no tuning stream, operation or original size) is rejected, not planned by the old families.
- **Inventory.** RK: OneWindow and DeepRSmallK (K <= 125: a 16-byte packet holds 4 FP32), AlignedK (4-, 8- and 16-byte
  packets; the 4-byte packet since 1228, at odd K, VALUE only), RuntimeK PacketComplete (tiles <= 124) and the rotated
  prefix (tile 125; no Exact layout). R / KR: Pack, Group, Block and Split, sharing FP8's kernels with a one-element
  word. Vector Pack was not ported: generic Pack reaches line rate on short FP32 rows.
- **ARG.** A two-word candidate: a 32-bit ordered key and a 64-bit tail (index << 1 | negative zero), compared key
  first. NaN and +-0 follow the public rules, and ARG runs at VALUE speed. Staged RK ARG writes packed 64-bit
  partials (key << 32 | shard row << 1 | -0) and finishes in the same pass.
- **Carried-index ARG.** Intermediates of any dtype's ARG plans carry FP32 values plus a UINT32 / UINT64 index. The
  composed index is not monotone in the row, so staged passes write wide partials (a key array, then a tail array at a
  256-byte offset) and merge with the full (key, tail) minimum. Contests time values and index together in one
  rotating input (`Fp32CarriedArgContestInput`).
- **Rankers and depths.** RK: model a37fea04 (VALUE and ARG, 1, 4, 96 and 192 MiB), depths {3, 8, 13}. KR: model
  787279aa (VALUE, ARG and carried ARG), depths {4, 4, 6}. Cache kinds `Fp32RK` and `Fp32KR`.
- **Misaligned input is not an error** (user, 2026-11-04). At an original-input site, the RK selectors keep only the
  passes that run at the input's alignment (`Tensor::alignmentBytes`, `fp32RKPassInputAlignment`). The first such
  input prints one warning per process. Every KR shape reads 4-byte words, so it runs at any FP32 address.
- **Results against the phase-0 audit (1176 to 1230).** ARG at KR and RK sites is 1.85-8.4x faster at 1-128 MiB
  (median per cell). VALUE at 16 MiB and up is unchanged or slightly faster. Small RK VALUE is 1.16-1.19x faster.
  Stamp time went from the old heuristic's 0.01 ms to a median of 5.5-20 ms by size.

### 8.2 What FP32 taught (rules for the next dtype)

- **Identities need their element width.** FP8 and FP32 plans of one geometry compared equal until
  `PhysicalKernel::element_bytes` (1178). FP16 and BF16 share a width, so their cache keys must carry the input dtype.
- **A cache key holds everything that changes the answer.** A carried-index final ARG KR site once shared a key
  across index dtypes (1215 / 1216).
- **Host references must match the GPU's NaN rules.** VALUE MIN / MAX propagate NaN, and `std::max` does not (1223).
- **Train where the latency is.** The first FP32 RK ranker saw only 96 and 192 MiB cells, and the audit's slow cells
  were 0.06-10 MiB. Sweep small working sets too (`--targets-mib=1,4,96,192`).
- **The old families can cover a regime the FP8 inventory lacks.** Odd K at small R had only cooperative passes until
  the 4-byte AlignedK packet (the retired KParallel's scalar mode) joined the pool (1228). Break audit regressions
  down by geometry (here, K parity) before tuning anything.
- **Continuations are tuned stages.** A staged candidate's continuation is chosen and cached per stage by the official
  selectors, so the first stamp of a multi-pass shape costs more (FP8 multi-pass: median 47 ms against 18) and later
  stamps reuse it (user decision 2026-11-04: accepted). Every caller that times staged candidates (sweeps, smokes)
  must give its selection context the original operation and reduction size (1229).
- **Route behind a switch, audit, then remove the switch.** The before / after audits ran on one build (1220-1227).
- **The old families stay until the last dtype that runs them converts** (user, 2026-11-04). FP32 no longer reaches
  `ReducersDenseR / KR / RK` or the direct ARG adapter, but their FP32 code is deleted with FP16 / BF16's conversion
  rather than cut out of shared kernels now (done in 1258-1261, section 9.3). Non-dense views (permuted, strided)
  keep their tiled and CUB paths for every dtype.

### 8.3 Open items left by FP32

- **R / KR VALUE below about 4 MiB.** 31 of 600 audit cells run 0.1-0.6 µs slower than the old CUB path (min 0.833),
  on Split (16) and Group (15). The FP32 KR ranker's cells already span 64 KiB-512 MiB, so unlike RK this is probably
  kernel latency (Split's separate finishing launch, Group at few rows), not training range. The next step is to time
  the FP32 KR pool against the old path at those cells.
- **Tests.** Phase 9's no-fallback and multi-step tests are `CubReductionFp32Routing.Random*` (randomized multi-pass
  VALUE and ARG plans on FP32 and FP8 inputs, every operation, the seed printed, every FP32-reading pass checked
  retained) and `CubDensePhysicalSelector.DescriptorOnlyFP32SitesGoToTheRetainedFp32Selectors`.

## 9. FP16 / BF16 (1233-1261)

FP16 and BF16 were the last dtypes through the pattern, and converted together: they share an element size and
differ only in decode and in their NaN / infinity encodings, like FP8's E4M3 / E5M2 pair. "F16" in names means both.
The plan, its measurements and its decisions are in the project's FP16 / BF16 plan
(`plans/Thor_FP16_BF16_Dense_Reduction_Plan_2026-11-04.md`).

### 9.1 What FP16 / BF16 have

- **Only original-input sites needed new work.** Every staged 2-byte pass writes FP32 partials, and every ARG
  intermediate carries FP32 values and an index, so the FP32 selectors already owned those sites.
  `selectDensePhysicalPass` sends original FP16 / BF16 sites to `F16RK::selectCached*Site` and
  `F16KR::selectCached*Site`; dense 2-byte ARG runs the planned ARG executor. A host-only context is rejected.
- **Inventory: the FP8 / FP32 kernels at element size 2**, as typed instantiations (one kernel source, a 2-byte branch
  where the mapping differs). RK: OneWindow and DeepRSmallK (K <= 249: a 16-byte packet holds 8 elements), AlignedK
  (2-, 4-, 8- and 16-byte packets x CTA 32 / 64 / 128 x Complete / Staged; the 2-byte packet at odd K only),
  RuntimeK PacketComplete (tiles 8..248) and the rotated prefix (tile 249, 8 cooperative warps, depths 32..1024).
  R / KR: Pack, Group, Block and Split, plus vector Pack (rows of 1-8 elements, VALUE only; user, 2026-11-05).
- **ARG.** R / KR ARG uses one 64-bit word: a format-neutral 16-bit ordered key << 48 | index << 1 | negative zero,
  merged with one unsigned minimum; FP16 and BF16 differ only in a runtime NaN threshold, so one kernel per geometry
  serves both. RK ARG widens each element to FP32's two-word key and reuses FP32's ARG kernels and packed partials.
- **Rankers and depths.** One model per site family for both dtypes, with a BF16 feature (FP8's E5M2 slot). KR:
  model e425a0b4 (VALUE and ARG), depths {4, 6, 8}. RK: model 3ad15d57 (VALUE and ARG, trained on the randomized
  15,538-cell sweep with a holdout), depths {4, 8, 12}; out of fold N=8 max 1.029 / 1.044 (VALUE / ARG), N=12 max
  1.016 / 1.020. Cache kinds `F16RK` and `F16KR`; keys hold the input dtype, so FP16 and BF16 never share an entry.
- **Misaligned input is not an error.** The RK selectors keep the passes that run at the input's alignment (at least
  2 bytes: rotated at 2, AlignedK at its packet), and the first such site warns once per process. R / KR runs at any
  2-byte address.
- **Results against the phase-0 audit (600 cells per dtype, p50 per-cell ratio).** KR ARG 2.6-15x, RK ARG 2.6-3.4x,
  RK VALUE 1.2-1.4x, KR VALUE 1.1-1.4x, R 1.0-1.3x. At 128 MiB and up every family runs at 1.64-1.69 TB/s median, and
  at 16 MiB and up no cell is slower than 0.98x. Per byte, 2-byte throughput is level with FP8's or about 1-2% above
  it at large sizes. Stamp time p50 7.4-24.6 ms by size (max 156 ms).
- **SASS (1259, sm_120):** FP16 / BF16 retained 11.75 MB (771 kernels), against FP8's 8.89 MB and FP32's 4.74 MB.
  RK AlignedK VALUE is the largest share (3.99 MB, 48 kernels): FP32's full configuration grid, against FP8's pruned
  one.

### 9.2 What FP16 / BF16 taught

- **A smoke sweep is not training data.** The RK ranker trains on the randomized sweep with a holdout, as every
  earlier ranker did, not on the short smoke run used to check the candidates (1251).
- **List order matters before the ranker exists.** Listing vector Pack first raised the heuristic list's worst first
  pick from 1.22x to 1.52x; second, it costs nothing and stays within the default depth (1235).
- **Share kernels through a type, and prove the other dtypes unchanged.** Every 2-byte instantiation of a shared
  FP8 / FP32 kernel came with a SASS size check that the existing instantiations were byte-identical.
- **Narrowing an entry point breaks callers by dtype, not by name.** When `launchKParallelTiledStage` became FP8 only
  (1259), an FP8-named test still ran its FP32 continuation through it (1260). Grep the callers and check the dtype
  each one passes.
- **Build-time generators need a run.** `fp8_selected_inventory_size.py`'s rotated cut marker matched twice from 1249
  to 1259 unnoticed, because `thor_tests` does not build that target.
- **Tests that assert a replaced path are pinned, then rewritten or deleted.** With the switch on, eight tests pinned
  the old routing (1255); the switch's removal rewrote five and deleted three (1257).

### 9.3 Cleanup (phase 10, 1258-1261)

- **1258:** the ordinary dense families (`ReducersDenseR / KR / RK`, `DenseRKInventory`, the direct dense plan) and
  their benchmark (`thor_cub_reduction_benchmark`). A dense site no retained selector owns throws. FP64 reductions are
  removed (user, 2026-11-05).
- **1259:** the legacy kernels: `CubReductionCooperativeShardedTiled.cu`, the generic FP16 / BF16 / FP32 KParallel
  grid (`launchKParallelTiledStage` is FP8 AlignedK only), the 8-warp rotated and FlatRows entry points, and the FP64
  code.
- **1261:** the interval-composition planner (VALUE and ARG), the composed and direct CUB ARG executors with their
  public API (`stampComposedDense*`, `analyzeDenseCompositionPlan*`), and `CubArgReductionMin / Max.cu`.
- Non-dense views (permuted, strided, rank-1 affine) keep their tiled and CUB paths for every dtype. Whether CUB stays
  for those is a separate decision, outside the dense pattern.

### 9.4 Open items left by FP16 / BF16

- **Small R / KR VALUE.** 20 FP16 and 24 BF16 audit cells below 0.95 (min 0.84), all VALUE below about 4 MiB and
  0.1-0.5 µs slower: Split's separate finishing launch, Group at R = 3, and a few small RK cells. This is FP32's gap
  (8.3), deferred with the Split cleanup.
- **The Split cleanup** (user decision 2026-11-05): make Split a staged Block pass with a planner-tuned continuation
  and the carried ARG format, for all three dtypes at once. The before data is measured (`split-before-{fp32,fp8}`).
- **2-byte AlignedK size.** Its 24 configurations per format are FP32's grid; FP8 compiles 9. Prune on sweep data,
  as FP8 did, if binary size matters.
- **Line-rate decisions were measured on GDDR (sm_120).** Generic Pack's short-row result, and FP8's and FP32's,
  may not hold on HBM parts, where each SM needs several times the bandwidth.

