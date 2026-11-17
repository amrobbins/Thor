# Thor reductions

Every GPU tensor reduction in Thor runs here: `Reducer` (sum, mean, product, min, max, L1 norm, L2 norm, sum of
squares), `ArgReducer` (argmin and argmax, with value, index or both), and the ragged `SegmentedReducer` /
`SegmentedArgReducer`. Expression, losses, metrics and the layers stamp these; nothing calls `cudnnReduceTensor` or a
CUB device reduction for a tensor axis. Every kernel is Thor-owned CUDA, chosen per pass for the pass's geometry on the
running GPU.

This document is the design as it stands: what each part does, the rules it keeps, and the measurements behind the
choices that are not obvious from the code. How a choice was reached is in the project's plans and logs. Retraining a
ranker for a GPU is in `benchmarks/FP8_RANKER_RETRAINING.md`.

## 1. Where things are

| Directory | Holds |
| --- | --- |
| `Reduction/` | The front end: `Reducer`, `ArgReducer`, `SegmentedReducer` (and its ARG twin), the stamps, `FrontEnd` (operation, dtype and view checks), `ReductionTuningDepth`, `ReductionTypes.h`, the benchmark hooks |
| `Layout/` | `ReductionGeometry` (the classification of a reduction's layout) and `ViewSplit` (how a caller runs a view the reducer refuses) |
| `Planner/` | The dense planner: the canonical problem (`DenseReductionPlanning`), the pass planner (`DenseReductionPlanner`), the per-site router (`DenseReductionPhysicalSelector`), its context, the one selection cache (`DenseReductionSelectionCache`) and device identity, the executable plan and pass execution |
| `Contest/` | The stamp-time contest every selector uses (`ReductionAutotuning::autotune`), its depth plan and the per-device GPU measurement lock |
| `Arg/` | The ordered ARG keys (`Fp8ArgKey.h`, `F16ArgKey.h`, `Fp32ArgKey.cuh`), the FP8 and FP32 ARG pass plans, the carried-index contest input |
| `KR/` | R / KR (contiguous rows): the shape vocabulary, pools and features (`KRSegments`), the pass plan; the selectors per dtype in `Fp8/`, `Fp32/`, `F16/` |
| `RK/` | RK (strided rows `[outer, R, K]`): physical identities, launch limits, per-family adapters, the RK pass plan, the FP8 cache-miss measurement (`RKSelectedContinuation`, `RKIsolatedSite`); the FP8 policies and selector in `Fp8/`, the FP32 and FP16 / BF16 candidates and selectors in `Fp32/`, `F16/` |
| `Kernels/` | The dense CUDA kernels by family (`KR.cu`, `RKOneWindow.cu`, `RKDeepRSmallK.cu`, `RKAlignedK.cu`, `RKRuntimeK.cu`), `StridedR.cu`, what they share (`KernelCommon.cuh`, `ReductionOperation.cuh`, `ScalarOps.cuh`, `StageSemantics.h`) and the launcher declarations (`KernelLaunchers.h`) |
| `Ragged/` | The ragged kernels (one object per family and dtype), their launchers, finishers, host / device walks and the ragged selector (`RaggedStampAutotuner`) |
| `Rankers/` | The generated ranker models (do not edit by hand) |

`Utilities/TensorOperations/Cub/` keeps only the CUB wrappers (sort, scan, select, top-k, run-length encode, find);
none of them reduces a tensor axis.

## 2. The contract

- **Dtypes.** Input and output storage: FP8 E4M3, FP8 E5M2, FP16, BF16, FP32. FP64 reductions are not supported.
- **Arithmetic.** Inputs convert to FP32; reduction state and finalization are FP32; the final store converts once to
  the output dtype. Non-final passes write FP32 (ARG: an FP32 value and a carried index).
- **Semantics.** The input transform (|x|, x^2) applies once, on the first pass. Finalization (Mean's divisor,
  L2Norm's square root), the runtime output scale and the output conversion apply once, on the last pass. ARG: NaN
  propagates, the lowest flattened index of the original reduction domain wins ties, and -0 ties +0.
- **Determinism.** No atomics. Multi-CTA work writes partials and finishes in a fixed order, so a stamp gives the
  same bits on every run. Between processes a site whose best shapes are within measurement noise may pick either;
  each is legal and equally fast.
- **Stamp, then run.** `stamp()` plans, tunes, selects and allocates every intermediate and scratch buffer. `run()` /
  `runOn()` launch the stamped passes and do nothing else: no planning, tuning, cache access or allocation.
- **Unsupported is an error.** A geometry, dtype or operation the kernels do not cover throws at stamp; nothing falls
  back to a generic or CUB path.
- **Rank.** Dense rank is dynamic; axis identifiers are `uint32_t`, and every executor carries 64-bit geometry.

## 3. Dense reductions

### 3.1 Lifecycle

```text
stamp:  current problem -> sites (R / KR / RK) -> physical selector -> one pass -> next problem -> ... -> plan
                                                      |
                                         selection cache (recipe only); on a miss:
                                         inventory -> ranker (top N) -> contest on this GPU
        then: allocate each FP32 intermediate and each pass's scratch
run:    the stamped passes, in order, on the caller's stream
```

### 3.2 The canonical problem and the planner

`DenseReductionPlanning` turns (dimensions, axes) into runs of kept and reduced extents, merging adjacent runs of the
same kind. A site is one reduced run, and its family follows from position: **R** (the whole tensor), **KR** (the
innermost run: contiguous rows), **RK** (a middle run: strided rows `[outer, R, K]`).

`planDenseReduction()` chooses one pass at a time. A pass reduces its site completely or partially; a partial
(staged) pass leaves an intermediate run, `[outer, shards, K]` in FP32, which is a site of the next problem like any
other. The planner analyzes every intermediate from scratch and chooses its pass as it would for a user's tensor.
So:

- no family is reserved for a stage number, and no architecture is a fixed `first stage + finisher` pair;
- stage role (first, intermediate, final) decides transforms, finalization and output scale, never geometry;
- workspace accounting covers every intermediate and every pass's scratch.

Sites are visited lazily, best bound first. A site's bound is the bytes of the next problem if one pass reduced it
fully, which no candidate for it can beat. Sites are visited in (bound, KR before RK, outer before inner) order and
selected (autotuned) only while the bound can still beat the best candidate so far. A site whose pass completes it
ends the step after one selection; a staged pass leaves more bytes and lets the next site be tried. The result equals
selecting every site, with fewer contests (`DenseReductionLazyPlannerTest`). The finished plan is a flat
`DenseExecutablePlan`, stamped as a whole.

### 3.3 Routing

`selectDensePhysicalPass()` sends every site to its dtype's selector, and there is no other policy:

| Site | Selector |
| --- | --- |
| FP32 input, and every FP32 intermediate of any dtype (VALUE) | `RK::Fp32::selectCachedValueSite`, `KR::Fp32::selectCachedValueSite` |
| FP32 ARG, original or carried-index intermediate of any dtype | `RK::Fp32::selectCachedArgSite` / `selectCachedCarriedArgSite`, the same in `KR::Fp32` |
| Original FP16 / BF16 input, VALUE and ARG | `RK::F16::selectCached*Site`, `KR::F16::selectCached*Site` |
| Original FP8 input, RK VALUE / ARG | `RK::Fp8::selectCachedIsolatedRKSite`, `RK::Fp8::selectCachedArgRKSite` |
| Original FP8 input, R / KR VALUE and ARG | `KR::Fp8::selectCached*Site` |

The context must carry the tuning stream, the original operation and the original reduction size; a host-only
context is rejected. Only a test's `proposal_override` plans without a selector. Dense ARG runs the same planner
(aggregate kind `Arg`, the carried index width from the whole original domain: UINT32 when it fits, else UINT64).
Each pass adds `local_run_index * domain_stride` to the carried index, so every legal pass order returns the same
public index.

### 3.4 The kernel inventory

The same families serve every dtype, templated on the input element, the transform and the combine:

| Family | Sites | Shapes |
| --- | --- | --- |
| OneWindow | RK | runtime K in buckets; O outputs per CTA, W warps, P packets; Complete |
| DeepRSmallK | RK | deep R, small K; staged by depth (FP32 partials `[outer, shards, K]`) |
| AlignedK | RK | K a multiple of the packet; Complete or staged by shard count |
| RuntimeK | RK | PacketComplete (tiled K, any address) and the rotated prefix layout; Complete or staged |
| Pack, vector Pack | R / KR | short rows (at most 32 bytes): one thread owns 2-16 rows |
| Group | R / KR | G lanes (1-32) per row |
| Block | R / KR | one CTA per whole row (rows of at least 1 KiB) |
| Split | R / KR | chunks of long rows with FP32 partials; a finishing kernel combines them in chunk order |

Per dtype:

- **FP8.** The full set. OneWindow is 15 runtime-K units (section 6.4); AlignedK compiles 9 configurations.
- **FP32.** OneWindow and DeepRSmallK at K <= 125 (a 16-byte packet holds 4 FP32), AlignedK at 4-, 8- and 16-byte
  packets (the 4-byte packet at odd K, VALUE only), PacketComplete (tiles up to 124) and the rotated prefix (tile
  125). R / KR: Pack, Group, Block, Split (generic Pack reaches line rate on short FP32 rows, so no vector Pack).
- **FP16 / BF16.** The FP8 / FP32 kernels at element size 2, as typed instantiations (a 2-byte branch where the
  mapping differs). OneWindow and DeepRSmallK at K <= 249, AlignedK at 2-, 4-, 8- and 16-byte packets x CTA
  32 / 64 / 128 x Complete / Staged (the 2-byte packet at odd K only), PacketComplete (tiles 8..248), the rotated
  prefix (tile 249, depths 32..1024). R / KR adds vector Pack (rows of 1-8 elements, VALUE only). "F16" in a name
  means both formats: they share an element size and differ only in decode and in their NaN / infinity encodings.

Every family has a host-side policy (`RK/Fp8/Fp8*Selector.h`, `RK/RKPhysicalIdentity.*`, the FP32 / F16 candidate
generators, `KR/KRSegments.*`) that answers, without CUDA, whether a shape is legal, how much scratch it needs and
which shapes exist (the pool). The compiled kernel answers residency (CTAs per SM of that exact specialization). Only
legal, resident shapes within the workspace budget become candidates. Physical identities carry the element width
(`PhysicalKernel::element_bytes`), and FP16 and BF16 keys carry the input dtype.

### 3.5 Rankers

One XGBoost model per dtype and site family orders the candidates; the contest times the top N. Each covers VALUE and
ARG, every operation and the dtype's formats, with the semantics as features. Features are computed in C++
(`rankerFeatures`, 26; `krRankerFeatures`, 24), and the training sweeps write the same features beside each timing,
so training never re-implements them. Training: `rank:ndcg` grouped by cell, labels grading regret against the cell's
fastest candidate (1%, 3%, 7%, 15%), folds that never split a shape, held-out shapes for validation only. The
exporter writes the trees as a C++ header evaluated in float32 in a fixed order, and parity fixtures that
`ModelMatchesTrainingFixturesExactly` checks bit for bit.

The ranker never decides alone: a weak ranker costs speed (the best shape falls outside the top N), never
correctness. Without a trained model a selector times its heuristic list's prefix.

| Model | Cells | Shortlist (Quick, Regular, Ultra) |
| --- | --- | --- |
| FP8 RK `b1a7bc53` | 15,614 at 1-256 MiB | 6 / 12 / 24 |
| FP8 KR `c82f0c6b` | 12,256 | 6 / 12 / 12 |
| FP32 RK `a37fea04` | 21,255 at 1, 4, 96, 192 MiB | 3 / 8 / 13 |
| FP32 KR `787279aa` | 24,512 (VALUE, ARG, carried ARG) | 4 / 4 / 6 |
| FP16 / BF16 RK `3ad15d57` | 15,538 at 1-256 MiB | 4 / 8 / 12 |
| FP16 / BF16 KR `e425a0b4` | 18,514 at 64 KiB-512 MiB | 4 / 6 / 8 |

The depths sit above the trainer's printed minimums for margin on GPUs and sites the sweeps did not see; each
header's comment gives its regret at each N. FP8 KR pools hold at most 15 shapes, so its Ultra stays at 12 and
differs from Regular by its rounds and trim.

### 3.6 The contest

`ReductionAutotuning::autotune` (`Contest/ReductionAutotuner.h`, `ReductionAutotunerCore.h`) is the one contest of
every reduction selector, dense and ragged. Its policy is the `ReductionAutotuning::Options` defaults; selectors set
only rounds and trim:

- **Cold L2.** Each launch reads its own slot of a ring sized for a reuse distance of 8 x L2. A slot is the input,
  output and scratch views together, 256-byte aligned (cudaMalloc's alignment), so every timed launch starts L2-cold
  and reads an input aligned as a real tensor's is. Nothing is flushed, and no allocation or copy is inside a timed
  interval. A staged candidate's continuation reads its fresh partials, as in a real run.
- **Probe, then batch.** After one warmup, every candidate gets one timed probe launch. Its launches per measurement
  are then clamp(ceil(200 us / probe), 1, 32): CUDA events resolve about 2 us, so short kernels batch until the timer
  is about 1% of the interval, and slow ones are not repeated.
- **Abandon the hopeless.** A candidate whose probe is over 3 x max(best probe, 8 us) is not measured and never
  wins.
- **Rounds and trim** (`reductionTuningPlan`): Quick 5 rounds (trim 1), Regular 7 (trim 1), Ultra 11 (trim 2),
  interleaved across candidates in a random order. The fastest and slowest `trim` rounds of each candidate are
  dropped, and the smallest trimmed mean wins (ties: the higher-ranked candidate).
- **One measurement at a time per GPU.** `ScopedReductionGpuMeasurement` serializes contests, and the allocation and
  release of internally owned representative inputs, on each device. Selection dependencies (an FP8 candidate's FP32
  continuation) are resolved before the lock is taken.

Why: with fixed launch counts a stamp's wall time was 4 ms plus its contest's GPU time (r = 0.9998), and below 16 MiB
92-98% of it went to shapes 10-100x slower than the winner (median 200-440 ms per FP8 KR stamp). With probes, batching
and abandonment the median stamp is 11-12 ms below 16 MiB, 19 ms at 16-128 MiB and 29 ms from 128 MiB, with picks as
good (mean 0%, p90 1%, max 4% over the sweep's fastest shape). With 32-byte ring slots a slot sat at a different
offset mod 256 from the next, and contest and sweep disagreed by up to 14%: hence 256-byte slots.

Contests time on an input of 1.0 everywhere; the kernels' speed does not depend on finite values.

### 3.7 Depth and the selection cache

- `ReductionTuningDepth` (Quick, Regular, Ultra) sets the shortlist and the rounds. A reduction's depth is explicit,
  else the network's (`Network::setReductionTuningDepth`, applied by `Network::place()`), else the calling thread's
  innermost `ScopedReductionTuningDepth`, else the process default (Regular unless set).
- `DenseReductionSelectionCache` is the one cache of every selector, dense and ragged: a bounded LRU of immutable
  recipes keyed by (kind, device identity, depth, site words), one kind per selector (`Fp8RK`, `Fp8KR`, `Fp32RK`,
  `Fp32KR`, `F16RK`, `F16KR`, `RaggedKR`, `RaggedRK`). A key holds everything that changes the shortlist, the
  measurement or the winner's legality (the input's base alignment included).
- Single flight per key; failures are not cached; `clear()` cannot be undone by a running flight and cannot
  invalidate a stamp. A deeper entry, completed or in flight, answers a shallower request.
- A recipe holds no Tensor, stream, executable or workspace. A hit costs no CUDA query and no allocation (device
  identity is cached per ordinal). The cache is not persisted on disk.

### 3.8 Stamped execution and workspace

Stamping allocates each non-final intermediate and each pass's scratch per executor; zero-scratch kernels allocate
nothing. `Reducer::queryWorkspaceSizeInBytes()` selects the same plan as stamping and sums those bytes (checked
addition), and the stamp checks its buffers against the sum. `getDensePhysicalPassDiagnostics()` describes the passes
(families, tensor geometries, dtypes, intermediate and scratch bytes) without inventing parameters a family does not
have. Workspace is the sum over passes; two ping-pong buffers would be enough (an open item).

### 3.9 Input alignment

A misaligned input is not an error. At an original-input site each selector keeps only the passes that read at the
input's base alignment (`Tensor::alignmentBytes`), and the first such site prints one warning per process. A pass
names the alignment it needs (`rkPassInputAlignment`, `fp32RKPassInputAlignment[AtK]`, `f16RKPassInputAlignmentAtK`,
`retainedFp8ArgPassInputAlignment`):

- RuntimeK PacketComplete (VALUE and ARG) reads any element address at any R and K: the kernel aligns its base down
  to a 16-byte packet and adds the offset to every row prefix. Every site therefore has a Complete pass at every
  alignment; below 16 bytes its tile may be narrower than a packet, the small-K job OneWindow does at 16 bytes.
- OneWindow (VALUE and ARG) takes its windows from the base the same way and reads any address when K fits a window
  that also holds a prefix (`oneWindowUnalignedCapacity`: K <= 497 / 241 / 113 / 49 at O1 / O2 / O4 / O8) and the
  input holds at most 2^32 - 16 bytes; 16 bytes otherwise.
- The rotated passes read any address; AlignedK needs its packet; DeepRSmallK needs 16 bytes.
- Every R / KR shape reads the input's 4-byte words on the tensor's word grid and masks the bytes outside its work, so
  it runs at any element address.

### 3.10 ARG

ARG reuses the VALUE inventory's geometry with its own kernels and an ordered integer key, so one unsigned comparison
orders values, NaN and +-0 as the public rules require, with no decode:

- **FP8** (`Fp8ArgKey.h`). Each byte maps to an 8-bit key: positive `x ^ 0x80`, negative `x ^ 0xFF`, -0 merged into
  +0's key, NaN key 0; four bytes per 32-bit word with plain integer operations. Argmax is argmin of the sign-flipped
  value and the NaN set a per-byte bias, both runtime, so one kernel per geometry serves argmin / argmax x E4M3 /
  E5M2. A lane candidate is one `uint32`, `key << 24 | row << 1 | negative_zero` (rows below 2^23 per lane, a kernel
  precondition), reduced with one unsigned minimum; across lanes it widens to `uint64`, `key << 56 | index << 1 |
  negative_zero`, indices up to 2^55 - 1. The value is rebuilt from the key, with no re-read.
- **FP16 / BF16** (`F16ArgKey.h`). R / KR: one 64-bit word, a format-neutral 16-bit key << 48 | index << 1 |
  negative zero, merged with one unsigned minimum; the formats differ only in a runtime NaN threshold. RK widens each
  element to FP32's key and reuses FP32's ARG kernels.
- **FP32** (`Fp32ArgKey.cuh`). A 32-bit ordered key and a 64-bit tail (index << 1 | negative zero), key first. ARG
  runs at about VALUE's speed.

The candidates at an RK site are the ARG twins of the VALUE MIN / MAX inventory (OneWindow, DeepRSmallK, AlignedK,
PacketComplete, rotated) plus the rotated RuntimeK ARG pass (depth min(R, 1024)), which is in every candidate set so
every site is covered (`retainedFp8ArgCandidates`, the FP32 / F16 ARG candidate generators). ARG ranks and autotunes
its own candidates with the shared model and the ARG aggregate feature. ARG lanes own at most 8 components: the key
update is about 8 SASS instructions per element and meets the load roofline at 4 or 8 components per lane, while 16
is schedule-sensitive and loses where warps are scarce. An ARG twin of a 16-component VALUE unit uses twice the lanes
under the same selector identity.

- **Packed staged partials.** Staged FP8 twins write one 4-byte lane candidate per partial (shard-relative row) into
  the pass's own scratch, and `fp8ArgPackedContinuationKernel` widens each with its shard's first row, takes the
  minimum and writes the requested outputs, in the same pass. To the planner a staged twin is one Complete pass with
  `outer * shards * K * 4` bytes of scratch (`retainedFp8ArgScratchBytes`). FP32 staged RK ARG packs
  `key << 32 | shard row << 1 | -0` the same way. (Writing an FP32 value + UINT32 index per partial moved twice
  VALUE's partial bytes.)
- **Carried index.** Intermediates of any dtype's ARG plan carry FP32 values plus a UINT32 / UINT64 index. The
  composed index is not monotone in the row, so carried staged passes write wide partials (a key array, then a tail
  array at a 256-byte offset) and merge with the full (key, tail) minimum. Contests time values and index together in
  one rotating input (`Fp32CarriedArgContestInput`).

### 3.11 FP8 RK cache-miss measurement

A staged FP8 RK candidate's speed depends on the FP32 passes that finish it, so on a cache miss each candidate is timed
as a complete program: `planIsolatedRKSite()` describes the isolated site, `selectIsolatedFp32Continuation()` chooses
each hypothetical FP32 pass through the same production selector (sequentially, before the measurement lock), and the
contest compares the complete programs. The winner contributes only its first FP8 pass to the cache; the benchmark
programs (`RKSelectedContinuation*`, `RKIsolatedSite`) die after measurement and are never a production executor. The
production planner then chooses each later pass from the actual next problem. A continuation is a tuned stage, cached
per stage, so the first stamp of a multi-pass shape costs more (FP8 multi-pass: median 47 ms against 18) and later
stamps reuse it. A caller that times staged candidates (sweeps, smokes) must give its selection context the original
operation and reduction size.

## 4. Views

`Reducer::analyzeGeometry()` classifies every input (`ReductionGeometry`). Canonical dense geometry resolves to
`SingleOutput`, `TrailingBlock`, `MiddleBlock` or `DisjointRuns`, and runs the dense planner. A view runs as follows
(the letters are the ones `FrontEnd.h` and `ViewSplit.h` use):

- **(A) every axis reduced** and **(B) the reduced axes the physical suffix**, with the kept axes in logical order:
  the dense planner on the view's bytes (`StridedSpan::asDense`), a dense `[input_elements]` or `[output_elements,
  reduction_size]`. The bits are the dense tensor's.
- **(D) natural-output permutation-aware views and (E) split physical reductions**: every compact permutation whose
  kept axes keep their physical order runs the dense planner on its bytes, its physical non-singleton axes collapsed
  into alternating kept / reduced runs (`viewPhysicalDenseProblem`). (Tiled kernels for (D) were 1.04-15,000x slower
  in every audit cell, worst with a few outputs of long reductions.)
- **(C) rank-1 strided views** (diagonals, broadcast aliases): StridedR (`Kernels/StridedReduction.h`), a stride-0
  closed form in FP64, else one Split launch on a grid that follows from n, whose last CTA combines the partials in a
  fixed order. Nothing is timed at stamp.
- **Refused: (F) rotated outputs, (G) the payload transpose, (H) pitched views.** `analyzeGeometry` throws
  `NotImplementedException`, and `Reducer::splitRefusedView` says how the caller runs them in two parts, a dense
  reduction and the dense copy (`copyToDenseAsync`). Expression does this itself (a DenseCopy stage).
  - F (a compact permutation `[O, R, I]` with output `[I, O]`) and G (`[A, R, B, P]` with output `[B, A, P]`):
    reduce the bytes as the dense `[A, R, B x P]`, then copy the buffer rotated (`RotatedOutput`). Rotating the
    reduced output costs 2 x output bytes per output against 2 x R x input bytes for rotating the input first.
  - H (rows of contiguous elements with gaps): reduce the view's dense copy (`PackedInput`); or, when the slabs have
    no gaps between them and the source holds the gap after the last row, reduce the rows read at their pitch as the
    dense `[O, R, Sr]` and copy the `[O, I]` columns (`SlicedOutput`, chosen when it moves fewer bytes:
    O R Sr b + O (Sr + 2 I) c against 3 O R I b + O I c). The gap columns are reduced and dropped.
  - Tiled kernels for these layouts ran 5-22,000x slower than the dense stamp of the same bytes.
- **Pitched views read in place.** A pitched view with enough outputs is read by one RuntimeK PacketComplete launch at
  its pitches (`Reducer::pitchedCompleteLaunch`, the strategy chain `pitched_complete:w<W>_t<tile>`), into the dense
  keepdims output, with no workspace and no copy. K tiles are balanced under the dtype's cap (124 / 248 / 496
  components) and at least 256 bytes; W is the most of 1 / 2 / 4 / 8 warps per `[outer, K tile]` item whose launch
  fits one wave of resident warps (32 per SM), with at least four rows per warp; in place when the launch keeps 4 KiB
  per SM in flight against a sliced split (which reads the view about once) or 2 KiB per SM against a packed one
  (three passes over the bytes). `analyzeValueSpanGeometry` and `readsSpanInPlace` accept such a span; a view the rule
  refuses packs or slices as above.
- **Anything else** (stride-2 payloads, zero-stride broadcasts at rank > 1, overlapping aliases, out-of-order split
  runs) throws `NotImplementedException` from structural analysis. There is no per-element logical-index backend.
  Einsum keeps repeated-label diagonal operands as views when a layout above owns their strides, and materializes the
  operand otherwise.

`ArgReducer::stamp()` requires a dense-contiguous input.

## 5. Ragged (offset-segmented) reductions

`SegmentedReducer` takes values `[N, D...]` and row offsets `[B + 1]` (UINT32 or UINT64) and produces `[B, D...]` for
sum, mean, min and max; `SegmentedArgReducer` returns global packed winner indices `row * D + component` (UINT32 or
UINT64) for argmin and argmax. Inputs are FP8 E4M3 / E5M2, FP16, BF16, FP32, accumulated in FP32. Empty segments give
0 (sum, mean), +inf (min), -inf (max) or the index sentinel, per component; NaNs propagate and the lowest packed index
wins ties. `stamp()` validates the offsets; `stampRuntimeOffsets()` (Expression's path) reads them only at run time.
Rows past `offsets[B]` (reserved capacity) are never read. A site no family runs is an error at stamp, with the
reason: rows over 2 GiB, or more than 2^30 components.

- **Partition.** The work is the merge of the offsets with the rows (each row weighted by its input bytes, each segment
  end by its output bytes), split evenly over a persistent grid fixed at stamp (SMs x resident CTAs). Each CTA finds
  its start with one binary search, so time depends on (active rows, B, D, dtype), not on the length distribution.
  That is also what makes a contest on synthetic offsets valid on the runtime-offsets path, where nothing knows the
  lengths at stamp. Shared host / device arithmetic: `RaggedMergePath.h`.
- **Families,** VALUE and ARG each, by row bytes: ragged KR (rows under 16 bytes: FP8 D 1-15, FP16 / BF16 1-7, FP32
  1-3; `RaggedKR*Kernels.cuh`, `RaggedKRWalk.h`) and ragged RK (rows of 16 bytes to 2 GiB: lanes own 16-byte chunks of
  a K tile of at most 512 bytes, wider rows K tiled; `RaggedRK*Kernels.cuh`, `RaggedRKWalk.h`). The row and K tile
  arithmetic is 32-bit, which sets the limits. ARG uses the dense keys (FP32 needs pair records). One kernel template
  per kernel for every input type, one object per family and dtype (`Ragged<Family>[Arg]<Dtype>[Narrow].cu`),
  launchers in `RaggedKR[Arg].cu` / `RaggedRK[Arg].cu`. Any element-aligned input address works.
- **Finishing.** Each CTA writes at most a head and a tail record for the segments it shares with its neighbours (the
  stamp's workspace); a second launch combines them in CTA order and stores the output (`RaggedFinish*.cu`: block,
  wide and FP32 pair-record finishers). Results are bit-identical run to run on one stamp; SUM may round differently
  between stamps with different grids. The finisher is a fixed cost of about 1.3-1.4 us, accepted against a
  single-kernel design that needs every CTA resident or an atomic ticket.
- **Selection** (`RaggedStampAutotuner`). Three candidate shapes per family, so every legal candidate is timed (no
  ranker, no depth: a depth selector earns its place only once the candidates more than triple), on uniform offsets
  over the stamped capacity, 9 rounds with one trimmed at each end. The first in list order within 1% of the best
  wins (the audit's p90 regret was 1.006-1.010). Winners are cached under `RaggedKR` / `RaggedRK`, keyed by operation,
  dtypes, D, capacity, B and the input's address mod 16. Pass names: `ragged_kr:<shape>` or `ragged_rk:<shape>`, then
  `ragged_finish` / `ragged_finish_arg`. `getPath()` stays `Segmented`.
- **Measured** (RTX 5090, copy about 1.54 TB/s, moved bytes >= 16 MiB, p50): 1.13 / 1.22 / 1.26 TB/s (E4M3 / FP16 /
  FP32); the spread between the fastest and slowest length distribution at one shape p90 1.07-1.31x. A
  one-owner-per-segment design ran at 0.03-0.06 TB/s with up to 3000x between distributions, though it suited
  balanced long segments, where the merge-path kernels are 0.83-0.85x of it at the median (an open item).

`RaggedExpression::segment_mean()` emits the direct segmented-mean stage (FP32 accumulation, division by row length,
empty rows and output conversion in one stage). Expression's segmented forward reductions and the segment min / max
backward (through `SegmentedArgReducer`) stamp this path; the backward zeroes the active prefix `[0, offsets[B] x D)`
on the device and scatters to the winners only, with no host readback.

## 6. Kernel notes

The measurements behind choices in the kernels. Each kernel file's comments hold the detail.

### 6.1 Work mapping and grids

DeepRSmallK, AlignedK and rotated RuntimeK divide nothing by a runtime extent on the device. They launch
`reductionWorkGrid(shards, component_tiles, outer)`: shard on `blockIdx.x` (fastest, so resident blocks stream
consecutive rows of one outer slab), tile on `y`, outer on `z`, each a 32-bit stride loop, so geometries past the
grid limits stay correct; normally each block owns one work item. Every extent is at most 2^31 - 1 so an index plus
its stride cannot wrap. AlignedK's balanced row split is computed once on the host, and every shard's row count fits a
32-bit counter. Row loops use 32-bit counters and a 64-bit row pointer.

Two measured exceptions keep a persistent 65535-block grid (`REDUCTION_PERSISTENT_MAX_BLOCKS`), where one block per
item was slower because launch and teardown outweigh the loads: AlignedK Complete items of at most 8 rows and 512
bytes (11%), and DeepRSmallK shards of at most 256 bytes (15%). RuntimeK PacketComplete keeps its persistent grid
(several W-warp outputs share a block) and its per-item split; a multiply-high division there measured 4-5% slower.

`RKLaunchLimits.h` restates each launcher's 32-bit limits for the host policies, so a stamp never ranks a kernel
that would throw: OneWindow outer x R x K <= 2^32 - 1; DeepRSmallK, AlignedK and rotated the work-grid extents;
AlignedK rows per shard <= 2^32 - 1; PacketComplete outer, K <= 2^32 - 1 and R <= 2^32 - 9 (its offsets are 64-bit,
so it runs inputs of 4 GiB and more). The packed ARG continuation takes R - 1 <= 2^55 - 1 and shards <= 2^31 - 1.

### 6.2 Register pins

- The FP8 rotated PrefixReserved kernel is pinned to three 512-thread blocks per SM (`__launch_bounds__(512, 3)`, at
  most 40 registers): at 41 or more it drops to two blocks and loses 15-23% on one-tile shapes.
- RuntimeK PacketComplete is software-pipelined (the next row's packet loads before the current row is staged) and
  pinned to four 256-thread blocks per SM: at most 56 registers (SUM was 62), 0.79x time on every operation.
- The DeepRSmallK tiny-K kernels sit at exactly 40 registers (6 blocks of 256 threads per SM).
- Rotated Exact keeps separate MIN and MAX kernels pinned to two blocks per SM at exactly 64 registers (the full
  MIN / MAX fold took it to 67, one block per SM, 1.47x slower).

Check `cuobjdump -res-usage` (REG, no LOCAL / STACK) after changing any of them.

### 6.3 Epilogues and folds

- **Complete epilogue.** nvcc hoists the launch-uniform output-dtype switch outside unrolled store loops, so an inline
  additive finalizer was copied once per store slot and dtype, each with its own divide and sqrt paths (about 31 KB of
  a 40 KB AlignedK Complete kernel). Complete kernels finalize through `finalizeFp32Epilogue()`: the identity case
  (divisor 1, no square root) is one inline uniform test, and MEAN and the square-root operations call
  `finalizeAdditiveFp32OutOfLine()`, one noinline copy per translation unit (the objects are not relocatable device
  code). OneWindow and PacketComplete compile no finalizer for PRODUCT / MIN / MAX. The AlignedK InBranch placement
  (P8 / CTA32, P8 / CTA64) keeps the inline finalizer: splitting it raised P8 / CTA32 from 79 to 82 registers and
  slowed identity shapes by up to 48%. The output store stays inline: a noinline finalize / scale / store saved 1.6 MB
  but took RuntimeK SUM from 62 to 80 registers (up to 1.9x slower).
- **FP8 MIN / MAX fold.** OneWindow and rotated PrefixReserved MAX launch the MIN kernel (`min.NaN.f32`), with the
  transform and the MAX sign applied to the packed FP8 bytes before decode, one LOP3 per four values
  (`Fp8ExtremumFold`): MIN identity `bytes`, MAX `bytes ^ 0x80`; ABS `bytes & 0x7f` (MAX: `^ 0x80` too); SQUARE as
  ABS, then squared. x^2 is monotone in |x| and FP8 squares are exact in FP32, so (max |x|)^2 equals max(x^2) bit for
  bit; a sign flip keeps NaN as NaN and swaps +-Inf and +-0, and `min.NaN` orders -0 < +0, so results match separate
  MAX kernels. OneWindow squares in-kernel and the host negates `output_scale` for MAX; the rotated first stage undoes
  the fold on each FP32 partial before storing it. PacketComplete is not folded (with compile-time transforms the fold
  only cost 1-2%).
- **Runtime FP8 format, rejected.** A format-agnostic OneWindow decode made it 1.5-2.3% slower (K >= 255 +4.5%) and
  cost 14 kernels per format a block per SM; it would have saved about 4.5 MB. The format stays a template axis.
  (ARG has no decode, so its format is runtime.)

### 6.4 OneWindow

One kernel, `oneWindowValueKernel`, with K a runtime operand bucketed by maximum components per lane (1, 2, 4, 8,
16), so `local[]` and the slot loop stay compile-time. Units (`ONE_WINDOW_UNITS`): P16 O1 c1, c2-la, c4,
c8-la, c16-la; O2 c1, c2, c4, c8-la; O4 c2, c4, c8-la; O8 c1, c2, c4; one loop form per (O, P, bucket), 90 FP8
kernels. The units were chosen by greedy oracle pruning (geometric mean <= 1.004 and per-cell <= 1.10 against all 36
P16 units). A K-specific kernel set had left models' K values outside its list on kernels built for other regimes at
0.15-0.9 TB/s; the runtime-K units match it where it won (0.995) and lift those K (median 1.46 to 1.50 TB/s, p10 0.47
to 0.97). The ten frozen variants (O, W, P) map to units (`launchFp8OneWindowForVariant`); P4 / P8 variants have no
unit (residency 0). `Fp8OneWindowPolicy::oneWindowCoversK` is K <= 497, or 512.

- The host computes rows per staged group, R workers per component and accumulating lanes
  (`OneWindowGeometry`). One-component buckets fold the R workers with butterfly shuffles when their
  count is a power of two (no shared-memory round trip); others keep the shared fold. The SUM worker fold is a tree.
- Decode: buckets 1 and 2 pack four rows of one component per FP8x4 decode; buckets 4-16 four of a lane's components
  of one row (packing rows there lost 1.14-3.6x).
- Load-ahead (`-la`) only where it improves the best configuration per shape with no shape slower than about 2%:
  pipelining every class was 0.91x overall but slowed bandwidth-bound shapes, often the ones a kernel wins.
- Rejected: 32-bit output indices (register spills past the 65 cliff, 1.12x slower at K 497); row-packed decode of a
  lane's partial last group (SASS 1.75 -> 3.84 MB, a block per SM lost).
- `Fp8OneWindowUnitsMatchReference` checks every unit against an exact CPU reference.

### 6.5 DeepRSmallK, AlignedK

- **DeepRSmallK depth** (`rows_per_shard`, one of `Fp8DeepRSmallKPolicy::compiled_rows`) is a runtime operand passed
  as log2, so the shard origin stays a shift (a 64-bit multiply cost 2-4% on shards of 4 KB or less). One kernel per
  (dtype, combine, transform, K ownership) serves all five depths (264 FP8 kernels instead of 840), except the
  one-component-per-lane kernel (K 9..32), compiled per depth (`FixedRowsPerShardLog2`): with the runtime depth it
  lost 4.8% at rows64.
- **DeepRSmallK prefetch.** Two or more components per lane (K > 32) prefetch the next row group's packet into a
  register before decoding the current group from shared memory (0.89-0.98x time). One-component and tiny-K kernels
  keep the plain loop (prefetching measured 1-5% slower).
- **DeepRSmallK decode.** Four or more components per lane decode four components of one row per
  `decodeFp8x4ToFp32` (c8 0.83-0.87x, c5 0.96-0.98x); one and two keep the scalar loop. Bit-identical.
- **AlignedK load-ahead** (`accumulateFp8AlignedKRows`): batches of 4 rows (P8) or 8 (P2) software-pipelined, FP8
  decoded in pairs; the hoisted Staged Identity P8 / CTA64 kernel loads 2 rows ahead (0.944x;
  `FP8_ALIGNED_K_HOISTED_STAGED_LOAD_AHEAD_ROWS`). Accumulation order is unchanged.

### 6.6 ARG kernels

- OneWindow twins (`fp8OneWindowArgKernel`, `oneWindowArgKernel`): one per unit, the VALUE unit's windows, staging,
  ownership and merge with a `uint32` candidate per owned component. With one or two components per lane four rows of
  one component reduce inside the key word (`fp8ArgRowsCandidate4`: four `__byte_perm`s, one minimum), used only where
  it measured faster (one component per lane, or load-ahead). One- and two-component units stage each packet linearly
  (O2 / O4 / O8 slices padded so subgroups never share a bank); c4 / c8 own quads of adjacent components (a key word
  is two LDS.32 and a funnel shift). O1 c16 (63 registers, next to the cliff) keeps lane + slot ownership.
- DeepRSmallK twins (7 ownerships): four or more components per lane pack four components of one row per key word,
  one and two pack rows; at K 9-16 floor(32 / K) workers share a component.
- Rotated RuntimeK ARG: 8-byte packets (`FP8_RUNTIME_K_ARG_PACKET_BYTES`; 16-byte packets were 1.56x slower).
- PacketComplete ARG: 16-byte packets, tiles up to 256 components (`fp8PacketCompleteArgTileComponents` keeps VALUE's
  tile unless a 256-component tile needs fewer key words per row).
- AlignedK twins (`fp8AlignedKArgKernel<2|8>`): VALUE's ownership, R loop and writeback (P8 through a per-warp shared
  transpose into 16-byte stores); the CTA size is a launch operand.
- Index-only FP8 ARG runs at about 1.03x VALUE overall (same-name OneWindow twins about 1.00x).

### 6.7 R / KR shapes

Every R / KR kernel reads 4-byte words and masks bytes outside its work. VALUE compiles only reachable semantics (SUM
x {identity, |x|, x^2}, PRODUCT, MIN / MAX through the MIN kernel with the FP8 fold). The heuristic list
(`fp8KRCandidateShapes`, and the FP32 / F16 lists by element size) gives the first guess and the pool
(`fp8KRCandidatePool`) adds the legal shapes around it; the ranker orders the pool:

- Pack first for short rows, Split first for few long rows; Split also for many rows of 16 KiB or more (chunks of 8,
  4, 16 and 32 KiB), where ARG Group lost 5-15% to it. Chunk counts sized to whole waves won 27 of 1048 cells and lost
  2% on average, so they are not used.
- Block first for rows of 2-64 KiB with at least one row per SM, where it won 81-100% of cells by 2-24% (Group wins
  below 2 KiB, Split from 64 KiB; with fewer rows than SMs Block leaves SMs idle). FP32 lists Block second for rows of
  64 KiB or more.
- The Split partial kernel runs at copy speed; its finishing launch (1.3-1.4 us) is the whole Split gap at 16-64 MiB
  (1.42-1.46 TB/s with it, 1.52-1.55 without). Removing it means in-kernel completion with an atomic ticket (an open
  item).
- FP16 / BF16 lists the 2-byte vector shape second (first, it raised the list's worst first pick from 1.22x to
  1.52x).

## 7. Rules

These are constraints. A new dtype, family or kernel keeps all of them.

1. **sm_80 and later.** No sm_90-only mechanism (PDL, TMA, clusters) in the production path.
2. **No atomics** in reduction kernels; fixed-order finishing.
3. **Every candidate is legal and compiled.** Selection never guesses a layout or occupancy: it asks the policy for
   legality and the compiled kernel for residency, and fails closed.
4. **The contest measures; the model only orders.** No model margin is read as time.
5. **One timing method.** The contest and the training sweeps measure the same way (cold inputs and outputs, probes,
   about 1 ms sweep samples or 200 us contest measurements). Changing the method means re-sweeping.
6. **No tuning, planning or allocation at run time.**
7. **Unsupported means an error**, never a silent fallback (`DenseFp8NoFallbackTest`, `ReducerFp32Routing.*`,
   `ReducerF16Routing.*`).
8. **Report bandwidth in TB/s** (moved bytes over time), measured reproducibly; a device copy of the same bytes is a
   secondary view where it helps (cells near the launch floor).
9. **CUDA builds with `--Werror all-warnings`.**
10. **Share kernels through a type, and prove the other dtypes unchanged.** A kernel whose other-dtype form differs
    only by the element type is a template shared across dtypes, and every new instantiation comes with a SASS check
    that the existing ones are byte-identical (`benchmarks/dense_reducer_sass_size.py --compare`). One whose shape
    logic differs is copied and lives independently.
11. **One ranker per dtype and site family**, so one dtype's kernels and rankers change without retraining another.
    FP8's E4M3 / E5M2 and FP16 / BF16 each count as one dtype.

## 8. Changing or adding a kernel or dtype

What is shared, and not to be copied: the planner and problem, the contest and the measurement lock, the selection
cache (add a `DenseReductionSelectionKind`, not a cache), the tuning depth and its scopes, the timing method
(`benchmarks/ColdTiming.h`) and the trainer, evaluator and exporter (`benchmarks/fp8_ranker/`, reused by
`benchmarks/fp8_kr_ranker/`).

The steps, in order:

1. **Audit first.** Time the current production stamps (R, KR and RK, VALUE and ARG) in TB/s with
   `thor_dense_production_audit --dtype=...` and `benchmarks/dense_production_audit.py`, and keep the run: it is the
   baseline every later step is compared with (`--compare`). Break regressions down by geometry (K parity, row
   length) before tuning anything.
2. **Kernels**, templated on the element, transform and combine; staged passes write FP32 partials; ARG writes its
   ordered keys. Transform once, finalize once, finish in a fixed order.
3. **Host policy**: legality, scratch and the pool, CUDA-free and unit-tested (`RKLaunchLimitsTest`,
   `Fp8KRRanker.PoolStartsWithTheListAndIsLegalAndDistinct`).
4. **Selector**: a `selectCached...Site` that builds the key (everything that changes the shortlist, the
   measurement or legality), ranks, contests through `ReductionAutotuning::autotune` and returns a recipe deriving
   `AcceleratorBackendSelectionRecipeTag`; route the dtype's sites to it in `DenseReductionPhysicalSelector`.
5. **Features and sweep**: C++ features, a training mode that writes them beside each timing. Train where the latency
   is: sweep small working sets too (`--targets-mib=1,4,96,192`), and train on the randomized sweep with a holdout,
   not a smoke run.
6. **Train, export, install**, with the parity fixtures and the retraining guide's audits.
7. **Tests**: a randomized multi-step test (`ReducerFp8MultiStepTest`: every operation, the seed printed, a few
   seconds a run), parity tests, depth and cache tests for the new kind. Host references must follow the GPU's NaN
   rules (`std::max` does not propagate NaN).
8. **Route behind a switch, audit, then remove the switch**; tests that pin the replaced path are rewritten or
   deleted with it.

Lessons that cost a patch each:

- An FP8 RK staged pass is finished by FP32 passes, and its ranker was trained on times that include them: when the
  FP32 path changes, re-run the FP8 audits (the retraining guide, steps 1 and 6) and retrain if the picks got worse.
- Narrowing an entry point breaks callers by dtype, not by name: grep the callers and check the dtype each passes.
- Build-time generators need a run: `thor_tests` does not build `thor_fp8_selected_inventory_size`, and a cut marker
  that matched twice went unnoticed.
- Limits are the kernels' integer widths, not round numbers; a site past them is an error, not a fallback.
- Remove a distribution axis before tuning: a contest is only as good as its stand-in for the real input.

## 9. Measuring

- **Batch launches.** CUDA events resolve about 2 us, so a 5-10 us kernel timed once sits on a 20-40% grid. Batch
  into about 1 ms samples (sweeps) or 200 us measurements (contests).
- **Rotate outputs too.** A shared output smaller than L2 hides write traffic (moved TB/s above the copy rate).
- **Time both sides the same way.** A production stamp against a sweep candidate timed differently disagreed by up to
  28%; a flush-per-sample audit disagreed with itself by as much.
- **Measure the hypothesis before fixing it.** Ring allocation looked like the stamp cost and alignment like a source
  of spread; a regression or an A/B disproved each.
- **Tools.** Sweeps: `thor_fp8_retained_cold_l2_sweep` (RK, `--training`, `--stamp`), `thor_fp8_kr_training_sweep`
  (KR; `fp8_kr_training.py` separates kernel gaps from list gaps and reports each depth's regret, `--compare` the
  run-to-run agreement). Audits: `thor_dense_production_audit` (any dtype), `fp8_rk_stamp.py`,
  `thor_ragged_reduction_audit` (`ragged_reduction_audit.py`). SASS: `dense_reducer_sass_size.py` (every reducer
  object by owner, family, kind and dtype; `--compare` holds each object and kernel to a previous run).
- **Inventories.** `thor_fp8_selected_inventory_size` compiles an isolated, selector-reachable FP8 inventory; it
  proves those specializations, not that production objects hold no others. `fp8_production_object_census.py` compares
  the production objects' exact mangled kernel identities with it per architecture. A production-only identity is not
  automatically dead code: trace its dispatch before deleting. Production object SASS, the selected inventory and the
  linked library size are different measurements; record the SM target and build settings.

## 10. Tests and guards

- Host-only: the physical identities, continuation layout, isolated site, RuntimeK proposals, production candidates,
  descriptor keys, the RK pass plan, the FP32 continuation chain, the measurement lock, the selection cache, the lazy
  planner, the launch limits, the ranker parity tests (`Fp8GeometryRankerTest`, `Fp8KRRankerTest`, `Fp32KRRankerTest`,
  `F16KRRankerTest`), the candidate generators (`Fp32RKCandidatesTest`, `F16RKCandidatesTest`), `RaggedMergePath`,
  `RaggedKRWalk`.
- GPU: the stamp contract (`Fp8RKStamp.*`), the numerics (`ReducerFp8MultiStep.*`; the FP8 RK, KR and ARG suites;
  `ReducerFp32*`, `ReducerF16*`, `ReducerElementWidthTest`), the routing and no-fallback tests, the view tests, the
  ragged suites (`RaggedKR*`, `RaggedSelector`, `RaggedRouting`, `SegmentedReducer*`, `SegmentedArgReducer*`,
  `RaggedExpression`), `ReductionTuningDepthTest`, `NetworkReductionTuningDepthTest`.
- Guards (`ExpressionReductionArchitecture.*`, `test/Utilities/Expression/ReductionSourceGuardTest.cpp`):
  - `ActiveSourcesDoNotUseCudnnReductionApis`: no cuDNN reduction descriptors, workspace queries or calls.
  - `GeneralReductionsAreCentralizedUnderCubReduction`: no direct general-purpose CUB value or ARG reduction in
    `Utilities`, `DeepLearning` or `bindings`.
  - `NoReductionCallsACubDeviceReduction`: no `cub::DeviceReduce::` or `cub::DeviceSegmentedReduce::` call and no
    device-reduce header anywhere, this directory included. The one exemption is `FlatScatterAddKernel`'s
    `ReduceByKey`, a keyed grouping rather than a tensor-axis reduction.
  - `ArgReductionDoesNotRetainSupersededKernelFamilies`, `SegmentedReductionDoesNotRetainLegacyOffsetSegmentedKernels`,
    `ValueReductionDoesNotRetainTiledViewFamily`, `ValueReductionDoesNotRetainAdaptiveCooperativeTiledKernel`,
    `ValueReductionDoesNotRetainLegacyLogicalIndexFallback`: the deleted kernels and the logical-index backend stay
    deleted.

## 11. Open items

- **Ping-pong workspace**: two scratch buffers instead of one per pass.
- **Split's finishing launch.** Small R / KR VALUE below about 4 MiB: 31 FP32 and 44 FP16 / BF16 audit cells run
  0.1-0.6 us slower than the reducer they replaced (min 0.83), on Split's separate finishing launch and Group at few
  rows. The plan: make Split a staged Block pass with a planner-tuned continuation and the carried ARG format, for all
  three dtypes at once (the before data is measured).
- **Pure R and ARG at 16-64 MiB** stay below 1.5 TB/s in some cells (the same finishing cost, occupancy-bound Group).
- **FP16 / BF16 AlignedK size.** Its 24 configurations per format are FP32's grid (3.99 MB, the largest share of the
  11.77 MB FP16 / BF16 SASS); FP8 compiles 9. Prune on sweep data if binary size matters.
- **Representative tuning input.** A realistic input would also exercise ARG's NaN and special-value paths.
- **Multi-GPU models.** One compiled model per ranker per build.
- **Measured on one GDDR GPU (sm_120).** Line-rate decisions (generic Pack on short rows among them) may not hold on
  HBM parts, where each SM needs several times the bandwidth.
- **Ragged:** balanced long segments (31-55 cells per dtype at >= 16 MiB below the one-owner design, worst 0.53x);
  a fused value + index for segment MIN / MAX (training reads the input twice); reusing the partitioner for segmented
  broadcast (the backward of `segment_sum`, `segment_softmax`).
