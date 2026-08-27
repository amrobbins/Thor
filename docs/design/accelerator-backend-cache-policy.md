# Accelerator backend cache ownership policy

Thor separates accelerator **selection state** from accelerator **execution state**.
The distinction is an ownership and reentrancy contract, not merely a cache-performance
choice.

## Policy

Process-global caches may contain only immutable compilation, tuning, or
algorithm-selection results. Cache keys are subject to the same rule: a safe selection
value does not make a descriptor-bearing key globally shareable. Independently executable operations must own their
backend descriptors, executable graph/plan objects, handles where applicable,
workspace, temporary storage, and any other potentially mutable or non-reentrant
runtime state.

In shorthand:

```
descriptor/configuration
        |
        v
process-global selection cache
        |
        +---- immutable algorithm / engine / knobs / workspace-byte requirement
        |
        +--------------------------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
          stamped operation A                 stamped operation B
          local executable A                  local executable B
          local workspace A                   local workspace B
```

There is no execution-state ownership edge between A and B. Once an operation has
been prepared/stamped, runtime execution must not need the process-global selection
cache and must not construct or tune a backend executable plan.

The universal safety property is **non-sharing/non-reentrancy**. A backend may impose
a stronger stream-affinity rule (cuDNN Frontend convolution currently does), but the
policy does not invent a universal "execute only on the construction stream" rule.

## Vocabulary

### Selection recipe: globally cacheable

A selection recipe is immutable after publication and may contain values such as:

- cuDNN Frontend engine id and knob choices, or an immutable serialized replay token when Frontend cannot losslessly expose a selected backend knob;
- cuBLASLt algorithm selection;
- algorithm or implementation identifiers;
- immutable tuning/heuristic facts;
- expected workspace or temporary-storage byte counts;
- immutable generated/compiled code artifacts when their backend contract permits
  process/device sharing.

`AcceleratorBackendSelectionRecipeTag` is the common C++ marker for backend-specific
selection value types introduced by later migrations.

### Local execution state: not globally cacheable

Operation-local execution state includes:

- cuDNN Frontend executable graphs/plans;
- cuDNN legacy tensor/filter/convolution descriptors;
- cuBLASLt operation and matrix-layout descriptors;
- descriptor-bearing `CublasKernel` state;
- backend handles unless the handle has a separate explicitly safe execution-domain
  owner (Thor streams already own their cuBLAS/cuBLASLt/cuDNN handles);
- workspace and temporary-storage allocations;
- counters, statistics, pointer attributes, or other state mutated by execution.

`AcceleratorBackendLocalExecutionStateTag` classifies backend-specific local executable
wrappers and causes defaulted copy operations to be deleted. Concrete executable
wrappers must also explicitly delete their own copy operations. Move-only ownership is
necessary but not sufficient: a global `shared_ptr` to that wrapper would still violate
this policy.

## Common cuDNN Frontend primitives

C2 introduces the canonical cuDNN Frontend implementation of this split:

- `CudnnFrontendPlanSelection` is the copyable immutable recipe: engine id, canonical
  sorted knob/value pairs, expected workspace bytes, and—only when Frontend's structured
  knob API is lossy—an immutable serialized replay token. The token contains bytes, not a
  live graph, descriptor, execution-plan object, handle, or workspace allocation.
- `CudnnFrontendPlanSelectionCache<Key>` is a bounded selection-only cache. Selection
  is single-flight per key, runs outside the cache mutex, and exposes entry/hit/miss
  diagnostics. `clear()` cannot be undone by a selection that started before the clear.
- `CudnnFrontendExecutablePlan` is the move-only operation-local wrapper. It privately
  owns the finalized Frontend graph, exposes no shareable graph pointer, and executes
  only that local graph.
- `cudnnFrontendPlanSelectionAtIndex()` extracts a deterministic recipe from a built
  candidate plan. If all reported knob identities can round-trip through Frontend, the
  recipe is structured `(engine, knobs)`. If Frontend collapses a backend knob to
  `KnobType_t::NOT_SET`, Thor stores Frontend's immutable serialized replay payload
  instead of rejecting every otherwise-valid primary plan. Frontend 1.27+ uses its
  plan-only serialization form; older supported Frontend releases may include immutable
  graph-structure bytes in the token, but still no live backend object.
- `replayCudnnFrontendExecutablePlan()` exact-replays a structured recipe on a pristine
  graph or deserializes the replay token into a fresh blank graph. Both happen during
  preparation only; runtime `execute()` never replays or deserializes. The reconstructed
  workspace requirement must match the cached selection before the local executable is
  returned.

These primitives do not by themselves create a process-global cuDNN selection cache;
individual backend migrations decide their descriptor key and instantiate the cache.

## Current Thor audit

C1 established the policy and C2 established the common cuDNN Frontend primitives.
C3-C11 migrated or removed the known execution-state caches. C12 deletes the transitional
cached-executable model and turns the repository audit into a hard gate: production source
may inventory known immutable selection-cache sites, but there is no allowlist for globally
shared backend execution state.

### cuDNN Frontend

No production cuDNN Frontend backend globally retains a finalized executable graph. The
old generic cached-execution-plan wrapper has been deleted; code must express either an
immutable selection recipe or operation-local executable ownership directly.

LayerNorm completed this migration in C3. Its process-global cache now contains only
`CudnnFrontendPlanSelection` recipes. Every `StampedLayerNorm` and every legacy
`LayerNorm` connection prepares a distinct move-only `CudnnLayerNormExecutablePlan`
and allocates its own workspace. Runtime execution consumes only that local plan;
clearing the global selection cache after preparation cannot invalidate an operation.

InstanceNorm completed the same end-to-end migration in C4. Its process-global state
contains only `CudnnFrontendPlanSelection` recipes. Each legacy `InstanceNorm`
connection prepares independent move-only forward/backward executables and allocates
independent workspaces during layer compilation. Runtime execution consumes only those
connection-local plans and may use another stream on the same GPU without consulting
the global selection cache.

AdaptiveLayerNorm completed the same migration in C5. Its process-global state contains
only immutable selection recipes, while each independently executable layer/application
owns its finalized forward/backward plan and workspace. Runtime cannot consult or
repopulate the selection cache.

RMSNorm completed the migration in C6. Dense legacy connections own private forward and
backward executables. `StampedRmsNorm` owns a complete local finite forward-plan family
for packed row capacities, and `StampedRmsNormBackward` owns its complete local backward
family plus a local fallback-forward family only when saved forward statistics are not
linked. Runtime row-capacity changes select only among these already-prepared local
plans. `warmForward()`/`warmBackward()` now warm immutable selections only; they do not
create a globally executable graph.

Attention/SDPA completed the migration in C7. Its global repository contains only
`CudnnFrontendPlanSelection` recipes. Attention intentionally uses a stronger placement-time
selection policy than the normalization wrappers: on a cache miss Thor asks cuDNN Frontend
for the Mode-B heuristic ranking, builds the first 16 successfully buildable candidates, and passes only
those built candidates to Frontend's empirical `autotune()`. The measured winner is cached
as an immutable serialized replay token because Frontend reorders its execution-plan vector
after tuning independently of the original heuristic engine-config vector. Candidate plans,
tuning buffers, and the tuning graph remain local to the cache-miss scope and are destroyed
after the winner is captured. Each `StampedAttention` owns a private forward executable and
workspace; if backward later requires retained stats, stamping replaces that local executable
with a local `generateStats=true` executable before runtime. Each `StampedAttentionBackward`
owns a private backward executable and, only when no matching forward state is linked, an
independent fallback-forward executable. Runtime attention execution binds tensors and
ragged metadata only; it never consults selection state, autotunes, or constructs/replays/
deserializes a Frontend executable.

Frontend convolution completed selection reuse in C11. Its process-global cache contains
only `CudnnFrontendPlanSelection` recipes, keyed by GPU/cuDNN identity plus operation kind,
tensor dtypes/dimensions, convolution geometry, grouping, compute dtype, and determinism
requirements. A cache miss performs the existing heuristic timing and then exact-replays
the fastest remaining candidate against Thor's independent full-output convolution oracle.
The recipe is not returned to the cache until that validation succeeds; rejected candidates
can never become globally reusable selections. The autotune graph and validation executable
are placement-local scratch state and are destroyed after selection.

Both cache misses and cache hits then exact-replay the immutable recipe into a fresh
`CudnnFrontendExecutablePlan` owned by that `BuiltConvolution`. Workspace size is checked
against the recipe during replay and workspace storage remains stamp-local. Runtime therefore
never consults or mutates the global selection cache and never shares a Frontend graph,
descriptor, execution plan, handle, or workspace across independently executable convolutions.

Expression softmax uses classic cuDNN with stamp-local tensor descriptors. `BuiltSoftmax`
is tagged local execution state and each stamped operation owns it exclusively.

### cuBLASLt

`knownHeuristicAlgorithms` already caches a selection value (`cublasLtMatmulAlgo_t`) and
is conceptually compatible with the policy.

C8 migrated the ordinary measured GEMM cache. `optimalKernelSelections` now stores only
copyable `CublasKernelSelection` recipes: the selected cuBLASLt algorithm/configuration,
workspace and wave metadata, plus immutable measured timing facts. `CublasKernel` is now
a move-only `AcceleratorBackendLocalExecutionStateTag` owner backed by `unique_ptr`; each
materialization creates its own cuBLASLt operation descriptors, matrix layouts, and local
`RunStats`. The existing tuning contest still uses local descriptor-bearing kernels, but
a winning kernel is snapshotted to a recipe before publication and the contest kernels
are then destroyed. Cache hits materialize a fresh local kernel from that recipe.

`knownHeuristicAlgorithms` remains a selection-only cache and is unchanged.

C9 completed the expression/bucketed/epilogue portion of the migration. The old global
`builtMatmulCache` is gone, and `BuiltMatmul` itself is now an operation-local execution-state
artifact held by `unique_ptr` from its `StampedMatmul`. Every stamped ordinary GEMM and
packed-row bucket family therefore retains its own descriptor-bearing `CublasKernel` execution
state. `LtMatmulPlan` is likewise a move-only `AcceleratorBackendLocalExecutionStateTag`
owner retained through a `unique_ptr`; its cuBLASLt operation descriptors and matrix layouts
can no longer be shared through a process-global expression cache.

Forward and backward epilogue contests now publish only copyable
`LtMatmulAlgorithmSelection` recipes into a process-global selection repository. Each stable
matmul key owns a preferred-workspace selection plus a preselected zero-workspace fallback;
workspace availability is not part of the key. The preferred contest uses a stable policy
cap derived from device capacity rather than instantaneous free memory. A caller-provided
workspace maximum is therefore a lookup constraint: Thor returns the preferred selection
when it fits, otherwise the zero-workspace fallback, without retuning or fragmenting the
cache. The repository uses per-key singleflight so expensive empirical contests for
unrelated shapes may proceed independently while equivalent concurrent stamps share one
selection effort. Cache hits materialize fresh local `LtMatmulPlan` descriptors. Clearing
the selection repository after stamping cannot invalidate or alter already-prepared plans,
and stamped runtime execution never consults or repopulates the selection cache.

### Classic cuDNN cleanup

Expression softmax uses classic cuDNN, but its tensor descriptors are now created per
stamped operation and retained only by that stamp's exclusively owned local `BuiltSoftmax`
execution state. There is no process-global softmax descriptor cache.

The old `GpuConvolution` production utility was removed in C10. It had no production
callers, while its cache key owned shared cuDNN descriptors; retaining or redesigning
that dead subsystem would only preserve a non-conforming ownership model. Convolution
tests that still need CPU reference geometry use a test-only pure-value requirement
object with no backend state.

### CUB

Thor's CUB plan structures are value-only operation metadata plus temporary-storage byte
requirements. Actual temporary storage is operation-local. No CUB ownership migration
is required; CUB is a useful known-good example of the intended split.

## Structural enforcement

C12 is the hardening gate. `AcceleratorBackendCachePolicyTest` scans production source for
forbidden global cache/container shapes containing cuDNN Frontend graphs, classic cuDNN
descriptors, cuDNN executable-plan wrappers, descriptor-bearing `CublasKernel` values, or
raw cuDNN/cuBLAS/cuBLASLt handles in process-style containers. It separately inventories
the explicitly approved immutable selection caches. Historical cache names such as the
expression MATMUL and softmax execution caches are also required to remain absent.

The post-migration audit removed the last dead alternate cuDNN-handle owner from
`CudnnHelper`; that helper now performs datatype conversion only. Backend library handles
remain owned by `Stream` state, which binds cuDNN/cuBLAS handles to its CUDA stream.

The RMSNorm+RHT+Amax helper is intentionally outside the cuDNN Frontend plan model: its
process-global cache stores only resolved CUDA launch metadata (`num_threads` and
`rows_per_cta`). Its public diagnostic is therefore named `cachedResolvedKernelCount()`
rather than `cachedGraphCount()` so the API does not imply that a backend graph is shared.

There is no execution-state debt allowlist after C12. A backend change that needs a new
process-global cache must make the cached value's immutable selection nature explicit and
extend the selection-cache inventory; descriptor/executable/workspace state remains local.

## Production ownership/runtime gate (C13)

C13 turns the ownership policy into executable regression coverage rather than relying only
on source shape.  Equivalent independently prepared operations must resolve to the same
immutable selection recipe while retaining distinct executable identities and, whenever
scratch is required, distinct workspace allocations.  The gate covers the migrated cuDNN
normalization families, SDPA, validated Frontend convolution replay, ordinary cuBLASLt,
bucketed cuBLASLt executable families, and cuBLASLt epilogue plans.

Every covered runtime path is also tested after its process-global selection cache has been
cleared.  Clearing selection state after stamping/preparation must not invalidate an
already prepared operation and execution must not repopulate that cache.  Monotonic
placement-time diagnostics make this mechanically observable:

- `cudnnFrontendExecutablePreparationCountForTests()` counts cuDNN Frontend replay/build/
  deserialize preparation and is unchanged by execution;
- `CublasKernel::materializationCountForTests()` counts descriptor-bearing ordinary or
  bucketed `CublasKernel` materializations and is unchanged by launch;
- `CublasMatrixMultiply::ltMatmulPlanBuildCountForTests()` counts local epilogue
  `LtMatmulPlan` builds and is unchanged by launch.

The concurrency gate prepares independent RMSNorm and attention branches, verifies equal
selection recipes plus pairwise-distinct executable/workspace identities, clears both
selection repositories, and then repeatedly executes the branches on independent CUDA
streams.  Convolution, ordinary matmul, and epilogue matmul ownership tests likewise run
independent stamped/local executables on separate streams after cache clear.  Correctness
checks are performed after the concurrent runs, so the gate catches both hidden shared
mutable state and runtime reconstruction of supposedly placement-local backend objects.

These counters and ownership diagnostics are testing surfaces only.  They expose identity
or monotonic construction facts, never backend descriptors/graphs for reuse, and therefore
do not weaken the selection-global / execution-local ownership boundary they verify.
