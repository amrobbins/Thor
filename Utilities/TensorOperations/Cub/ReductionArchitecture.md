# Thor Reduction Architecture

Thor's central GPU reduction implementation lives in `Utilities/TensorOperations/Cub`.
Dense expression reductions and ragged offset-segmented reductions use these utilities directly; they do not stage
inputs through compatibility tensors and do not call `cudnnReduceTensor`.

## Numeric contract

- Input storage may be FP8 E4M3, FP8 E5M2, FP16, BF16, FP32, or an enabled FP64 type.
- Input iterators convert values to FP32 before reduction.
- Reduction state and operation-specific finalization use FP32.
- The final store converts once to the configured output storage dtype.
- Stamped operations own their output and CUB workspace. `run()` and `runOn()` do not allocate or re-plan.

## Dense paths

`CubReduction` and `CubArgReduction` select one of three paths during stamping:

1. Device transform-reduce when the result contains one element.
2. Fixed-size segmented reduction when the reduced axes form a contiguous suffix.
3. Fixed-size segmented reduction over a logical counting/transform iterator for leading, middle, or disjoint axes.

Dense reduction rank is dynamic. Strided-path dimensions, strides, and axis lists are packed into a rank-sized GPU
metadata tensor while stamping and referenced by the transform iterator at execution. There is no cuDNN-derived rank-8
limit; the only representation bound is that axis identifiers are `uint32_t`.

Value reductions support sum, product, mean, min, max, L1 norm, and L2 norm. Arg reductions produce deterministic
local flattened indices: NaNs propagate, and the lowest logical index wins equal-value ties.

## Offset-segmented path

`CubSegmentedReduction` owns ragged sum, mean, min, and max. `RaggedExpression::segment_mean()` emits the direct
segmented-mean stage, so FP32 accumulation, division by row length, empty-row handling, and output conversion happen in
one CUB operation without materializing row lengths or segmented sums. Segment offsets are validated while stamping,
and empty segments use the explicit identities defined by `CubReduction::getFp32EmptyReductionValue()`.

## Expression integration

`BuiltReduction` caches the normalized axes, result kind, operation, and geometry. A plan produces either a value or
indices; the backend is not selectable. `StampedReduction`, `StampedArgMinMax`, and
`StampedReduceMinMaxBackward` bind those plans to concrete tensors. Min/max backward retains Thor's scatter kernel
after CUB computes the winning local indices.

The test `ExpressionReductionArchitecture.ActiveSourcesDoNotUseCudnnReductionApis` prevents the retired cuDNN
reduction descriptors, workspace queries, and execution API from being reintroduced anywhere in active Thor sources.
The source guard `ExpressionReductionArchitecture.GeneralReductionsAreCentralizedUnderCubReduction` scans Thor's
`Utilities`, `DeepLearning`, and `bindings` sources for direct general-purpose CUB value or arg reductions. The obsolete
standalone `CubDeviceReduce*` and `CubDeviceSegmentedReduce*` primitive wrappers were removed after all value and
offset-segmented callers moved to the central utility. `FlatScatterAddKernel` still uses CUB ReduceByKey,
which is a keyed grouping primitive rather than a tensor-axis reduction and therefore remains separate.

Loss shaping uses central CUB sums with an explicit FP32 output scale: batch and classwise losses divide only by the
batch size, while elementwise losses sum non-batch elements without normalization. Binary accuracy uses the same
scaled-sum facility. The obsolete `BatchReduce` class has been removed.
