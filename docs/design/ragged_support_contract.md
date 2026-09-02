# Rank-1 ragged support contract

This document is the repository-level contract and support matrix for Thor's
first-class ragged execution surface. It describes what is supported today,
which restrictions are intentional, and which dense operations are not yet
part of the ragged API.

R1 is a qualification/documentation patch. It does not add new numerical
behavior. Future ragged patches should update this matrix and extend the
`check-ragged-support-contract` gate when they promote a capability to
first-class support.

## Scope

Thor's public `RaggedTensor` is a **canonical rank-1 sequence-ragged tensor**:

```text
RaggedTensor
  values:  [max_total_values, trailing...]
  offsets: [batch_size + 1]
```

The required structural invariants are:

- `offsets[0] == 0`;
- offsets are monotonically non-decreasing;
- `offsets[batch_size] <= max_total_values`;
- `offsets` uses `UINT32` or `UINT64` according to the descriptor;
- the logical packed extent is exactly `[0, offsets[batch_size])`;
- `[offsets[batch_size], max_total_values)` is inactive capacity with undefined
  contents;
- `max_values_per_row`, when present, is placement-time capacity metadata and
  every logical row length must fit it;
- runtime caches such as the host active-value count are derived from the
  canonical offsets allocation and are never serialized as model state.

Nested ragged ranks are outside this contract. A future nested/jagged tensor
abstraction must not silently change the meaning of the existing rank-1 type.

The physical ownership and consumer-responsibility rules are specified in
[`ragged_row_partition_runtime.md`](ragged_row_partition_runtime.md).

## Operation classes

A **partition-preserving** operation transforms values but retains the exact
canonical offsets tensor:

```text
(values A, offsets P) -> operation -> (values B, offsets P)
```

A **partition-changing** operation changes row membership or segmentation and
must explicitly produce a new canonical offsets tensor:

```text
(values A, offsets P) -> repartition -> (values B, offsets Q)
```

Using `RaggedTensor::withValues()` is valid only for the first class. There is
no implicit mechanism by which a values tensor can manufacture a new row
partition.

## Public support matrix

Status meanings:

- **Supported**: public C++ API exists and the capability is covered by ragged
  regression tests.
- **Restricted**: public ragged API exists, but only the listed geometry or
  configuration is part of the contract.
- **Structural**: consumes row-partition metadata and returns dense structural
  information rather than a ragged values tensor.
- **Not public**: lower-level machinery may exist, but there is no first-class
  public ragged layer contract yet.
- **Out of scope**: the dense operation does not currently have an agreed
  rank-1 ragged meaning.

| Surface | Status | Rank-1 ragged contract / restriction | Primary regression evidence |
| --- | --- | --- | --- |
| `RaggedNetworkInput` / `RaggedNetworkOutput` | Supported | Logical boundary is values + authoritative offsets; `partition=<existing ragged input>` declares a values-only external stream sharing the exact existing partition instead of duplicating structural metadata/offset ports; inactive capacity remains undefined | `RaggedNetworkInputApi.*`, `RaggedNetworkOutputApi.*` |
| `RaggedTensor` serialization / descriptors | Supported | `UINT32` and `UINT64`; optional `max_values_per_row`; runtime extent/cache is not serialized | `RaggedTensorApi.*`, implementation descriptor/runtime tests |
| `NumpyDataset` host/device residency | Supported | Canonical packed values + offsets materialize through both host-backed batching and generic device-resident snapshots; STRICT device storage supports dense/ragged mixtures and enforces requested `max_total_values` / `max_values_per_row` bounds | Python `test_numpy_dataset_ragged_batches_train_through_canonical_ctc_with_exact_partial_tail`, device-resident named-session capacity tests |
| Ordinary standalone activations | Supported | Tokenwise over active packed values and partition-preserving | `Activations.ShapePreservingBuildersInferRaggedFromInputAndPreserveOffsets` |
| GLU-family activations | Supported | Tokenwise; final trailing feature dimension must satisfy the gate/split geometry | `GatedLinearUnits.Ragged*` |
| `Softmax` | Not public | Ordinary dense/channel softmax remains distinct from sequence-axis ragged normalization and does not acquire implicit segmented semantics | `Softmax::supportsRaggedStandalone()` guard |
| `SegmentedSoftmax` / `SegmentedLogSoftmax` | Supported | Normalize independently across each variable-length row for every trailing component; exact partition preservation; FP16/BF16/FP32 values only, with FP64 intentionally unsupported | `SegmentedPrimitiveApi.*`, Python `test_segmented_primitives.py`, `RaggedExpression.*SegmentSoftmax*` |
| `FullyConnected` | Restricted | Tokenwise final-dimension projection; prefix preservation required; ordinary supported activation contract applies; custom epilogues are supported; auxiliary bindings must be same-partition `RaggedTensor` inputs and are active-prefix-aware | `FullyConnectedApi.Ragged*`, Python `test_fully_connected.py` |
| `RMSNorm` | Restricted | Normalization axes must stay within trailing value dimensions; custom epilogues are supported; auxiliary bindings must be same-partition `RaggedTensor` inputs and remain active-prefix-aware | `UtilityApiLayers.RaggedRMSNorm*`, Python `test_rms_norm.py` |
| `LayerNorm` | Restricted | Public ragged LayerNorm is token-wise over exactly one non-zero trailing channel dimension; `normalizedShape` must equal that trailing dimension. Multi-axis ragged normalization remains outside the current expression backend contract | `UtilityApiLayers.RaggedLayerNorm*`, Python `test_layer_norm.py`, `RaggedExpression.*LayerNorm*` |
| `DropOut` | Supported | Training touches only the logical extent; inference/validation identity preserves the partition | `DropOut.Ragged*`, `UtilityApiLayers.RaggedDropOut*` |
| `Add` | Restricted | Both operands must be ragged with the exact same canonical offsets tensor and compatible values descriptors | ragged `CustomLayer`/transformer integration coverage |
| `Concatenate` | Restricted | Concatenates trailing feature axes only; every input must share the exact same partition | `UtilityApiLayers.RaggedConcatenate*` |
| `Slice` | Restricted | Slices trailing value axes only; does not slice/repartition the sequence axis | `SegmentedReductionApi.RaggedSlice*`, `RaggedExpression.TrailingSlice*` |
| `TypeConverter` | Supported | Active-prefix tokenwise conversion; partition-preserving | `UtilityApiLayers.TypeConverterRagged*` |
| `RaggedRowLengths` | Structural | Materializes row lengths from canonical offsets as dense metadata | Python `test_ragged_add_and_row_lengths.py`, `TrainingRunsResult.Save*Ragged*` |
| `SegmentedReduction` | Supported | `SUM`, `MEAN`, `MIN`, `MAX` reduce each row to dense per-row output | `SegmentedReductionApi.*`, `RaggedExpression.*Segment*` |
| `SegmentedBroadcast` | Supported | Dense per-row -> ragged per-token broadcast using only the partition input's offsets; exact partition preservation; broadcast values FP16/BF16/FP32 only, with FP64 intentionally unsupported | `SegmentedPrimitiveApi.*`, Python `test_segmented_primitives.py`, `RaggedExpression.SegmentSumAndMeanAutodiffLowerThroughSegmentedBroadcast` |
| `CustomLayer` | Restricted | Named ragged inputs must share one canonical partition; output is partition-preserving; conditional/full-batch semantics remain restricted | `RaggedCustomLayer.*` |
| `Attention` | Supported | Ragged Q/ragged KV and both mixed dense/ragged Q/KV quadrants are public; ragged additive score-bias backward remains backend-limited | `AttentionApi.*Ragged*`, `CudnnMixedRaggedAttention.*`, `RaggedTransformerCompleteness.*` |
| `ScaledDotProductAttention` | Supported | Query and key/value domains may be dense or ragged independently; key/value must share a domain and exact partition when ragged; mixed/ragged execution uses BSHD and a private uniform partition for the dense side; output is ragged iff query is ragged; ragged additive-bias backward remains backend-limited | `AttentionApi.Sdpa*Ragged*` |
| causal `Convolution1d` | Restricted | Rank-1 `[N,C]`, stride 1, causal padding, finite placement-time `max_values_per_row`; grouped/depthwise and retained backward are supported | `RaggedConvolution1dPublicIntegration.*`, T9/T10 retained-ragged gates |
| `CtcLoss` ragged labels | Supported | Ragged labels are first-class; CTC's label semantics are specialized and do not define the reduction policy for ordinary losses | `CtcRaggedArchitecture.*`, `RaggedCtcEndToEnd.*` |
| ordinary losses / metrics | Restricted | `MAE`, `MSE`, `MeanPowerError`, `MAPE`, `HuberLoss`, `SmoothL1Loss`, `QuantileLoss`, `ExpectileLoss`, `AsymmetricPowerLoss`, `BinaryCrossEntropy`, `BinaryFocalLoss`, `PoissonNLLLoss`, `TweedieLoss`, `GammaNLLLoss`, `GaussianNLLLoss`, and `LaplaceNLLLoss` are public ordinary ragged valuewise losses. Predictions/labels must share the exact partition and each loss keeps the same value dtype contract as its dense counterpart. Gamma's optional differentiable dispersion, Gaussian's variance, and Laplace's scale are also ragged differentiable inputs and must share that exact partition and value geometry. Dense `[1]` example weights are supported wherever the corresponding dense loss exposes weights and are segmented-broadcast to every active token in the corresponding logical row; losses that do not expose dense example weights do not invent ragged-only weighting. `RAW` preserves the prediction partition, `PER_EXAMPLE` is a dense segmented row sum, `BATCH` averages row sums over valid logical examples (not by active-token count; weighted losses also do not divide by weight sum), `NONE` remains a training-only root, and `PER_OUTPUT` is rejected. Ragged MAPE preserves the dense stability contract (`epsilon=1e-4`, maximum loss/gradient magnitude 1000). Other ordinary loss/metric families remain pending. | `RaggedMAEApi.*`, `RaggedRegressionR10E.*`, `RaggedRegressionR10F.*`, `RaggedRegressionR10G.*`, `RaggedClassificationR10H.*`, `RaggedDistributionR10I.*`, `RaggedDistributionR10J.*`, Python regression/classification/distribution loss tests; common execution/shaping in `RaggedCustomLoss.*` / `RaggedLossShaper.*` |
| `Embedding` | Supported | Integer ragged indices map tokenwise to floating embeddings while preserving the exact canonical partition; forward and sparse backward are bounded by `offsets[B]`, never inspect inactive packed index/gradient capacity, reuse captured sparse-update graphs across changing active extents, and support FP16/BF16/FP32 embedding values with UINT8/16/32/64 indices and UINT32/64 offsets | `EmbeddingApi.Ragged*`, `EmbeddingRaggedRuntimeTest.*`, Python `test_embedding.py` |
| `FiniteCheck` | Supported | Zero-copy diagnostic identity; forward/backward inspect only the authoritative active packed prefix ending at `offsets[B]`; inactive capacity remains undefined and is ignored | `UtilityApiLayers.RaggedFiniteCheck*`, `FiniteCheck.Ragged*`, Python `test_finite_check.py` |
| `StopGradient` / `ScaleGradient` | Supported | Forward preserves the exact canonical offsets object; gradient control applies only to packed values and never changes row membership | `UtilityApiLayers.RaggedStopGradient*`, `UtilityApiLayers.RaggedScaleGradient*`, Python gradient-control tests |
| trailing `Reshape` / `Flatten` | Supported | Metadata-only transforms of each packed value's trailing shape; element count per packed value and the exact canonical row partition are preserved | `UtilityApiLayers.RaggedReshape*`, `UtilityApiLayers.RaggedFlatten*`, Python `test_ragged_shape_ops.py` |
| trailing `Transpose` | Restricted | Swaps only the final two trailing value dimensions; the packed row axis is never transposed. Materialization/backward uses an active-prefix non-overlapping strided view | `UtilityApiLayers.RaggedTranspose*`, `RaggedExpression.TrailingTranspose*`, Python `test_ragged_shape_ops.py` |
| `RaggedSequenceConcatenate` | Supported | Concatenates corresponding rows in input order across independently partitioned rank-1 ragged inputs; same batch/values dtype/offsets dtype/trailing shape required; explicitly produces a new canonical offsets tensor `Q[row] = sum_i P_i[row]`; forward/backward touch only active packed values | `UtilityApiLayers.RaggedSequenceConcatenate*`, `RaggedSequenceConcatenate.*`, Python `test_ragged_sequence_concatenate.py` |
| `RaggedSequenceSlice` | Supported | Applies a fixed non-negative `start` and positive `length` independently to each logical row, clips short rows, compacts selected active values, and explicitly produces a new canonical offsets tensor; backward zeros sliced-out active source gradients and scatters selected gradients without touching inactive capacity | `UtilityApiLayers.RaggedSequenceSlice*`, `RaggedSequenceSlice.*`, Python `test_ragged_sequence_slice.py` |
| `RaggedGather` | Supported | Interprets scalar UINT32/UINT64 indices row-locally against source partition P; output values use source dtype/trailing geometry and reuse the indices partition Q exactly; source and indices offsets dtypes may differ; duplicate indices accumulate during backward; inactive capacities are ignored | `UtilityApiLayers.RaggedGather*`, `RaggedGather.*`, Python `test_ragged_gather.py` |
| `RaggedFilter` | Supported | Stable row-local filtering with one scalar BOOLEAN predicate per active token; mask and values must share the exact canonical partition; selected tokens are compacted in order into a fresh partition; forward/backward ignore inactive capacity and mask values are non-differentiable | `UtilityApiLayers.RaggedFilter*`, `RaggedFilter.*`, Python `test_ragged_filter.py` |
| `RaggedToPaddedDense` | Supported | Losslessly materializes canonical ragged rows as ordinary dense `[B,W,...]` storage using the declared finite `max_values_per_row=W`; inactive packed capacity is ignored, short rows are filled with an explicit constant padding value, and backward discards padding gradients | `UtilityApiLayers.RaggedDenseAdapters*`, `RaggedDenseAdapters.*`, Python `test_ragged_dense_adapters.py` |
| `PaddedDenseToRagged` | Supported | Packs ordinary dense `[B,W,...]` storage according to an existing canonical `partition_input`; only its offsets are consumed, `W` must cover `max_values_per_row`, output reuses the exact partition object, padding cells are ignored, and backward emits exact-zero padded gradients | `UtilityApiLayers.RaggedDenseAdapters*`, `RaggedDenseAdapters.*`, Python `test_ragged_dense_adapters.py` |
| `AdaptiveLayerNorm` | Restricted | Rank-1 `[N,C]` ragged data uses dense per-logical-row `[C]` scale/bias inputs; `SegmentedBroadcast` expands each row's conditioning only across its active tokens, the exact partition is preserved, and normalization reuses packed finite-bucket `LayerNorm`. Multi-axis ragged normalization remains out of scope and training inherits the current packed `LayerNorm` autodiff gate | `UtilityApiLayers.RaggedAdaptiveLayerNorm*`, Python `test_ragged_adaptive_layer_norm_*` |
| `BatchNorm`, `InstanceNorm`, Conv2D/3D, 2D pooling, unrestricted `Einsum` | Out of scope | A single unambiguous rank-1 sequence-ragged meaning has not been standardized | dense-only public surfaces |

## Qualification gate

The repository-level C++ gate is:

```bash
cmake --build <build-dir> --target check-ragged-support-contract
```

The target deliberately runs a disabled CUDA preflight. Individual CUDA tests
may skip on a machine without a GPU during ordinary development, but the
qualification target must not pass vacuously without exercising the CUDA-backed
ragged paths.

The gate gathers rather than duplicates the main numerical tests. In
particular it covers:

- canonical API/descriptors and both offset widths;
- logical network input/output boundary behavior;
- NaN-poisoned inactive capacity;
- all-empty partitions;
- short -> long -> short reuse of one placed network;
- save/load followed by a different runtime partition;
- architecture save/load and subgraph clone wiring;
- active-prefix forward/backward behavior in FullyConnected, RMSNorm, LayerNorm,
  DropOut, Embedding, and FiniteCheck;
- partition-preserving gradient-control behavior in StopGradient and ScaleGradient;
- tokenwise activations, type conversion, trailing concatenate/slice/reshape/flatten/transpose,
  segmented reductions/broadcast/softmax/log-softmax, ragged CustomLayer behavior, and explicit
  ragged <-> padded-dense round trips;
- ragged and mixed dense/ragged attention;
- ragged CTC label/training integration;
- public causal Conv1D integration. The heavier retained-backward and timing
  qualification remains separately enforced by
  `check-retained-ragged-training-production-gate`.

The boundary-sequence cases live in `RaggedSupportContract.*`; layer-specific
numerical details remain in their natural test files.

Python API parity should be checked with the corresponding focused tests. The
Python data tests also qualify host-backed and STRICT device-resident ragged
`NumpyDataset` materialization. For example:

```bash
pytest -q bindings/python/test/core/layers \
  -k 'ragged or segmented'
```

The Python suite is not invoked by the CMake C++ gate because a configured
Python test environment is not a prerequisite for building the C++ library.

## Required regression pattern for new ragged operations

A new first-class ragged operation should not be marked **Supported** until its
tests address the applicable parts of this checklist:

1. **Structure**: output offsets are exactly preserved for a
   partition-preserving operation, or a new canonical partition is explicitly
   produced for a partition-changing operation.
2. **Offset width**: `UINT32` and `UINT64` offsets either both work or the public
   builder rejects the unsupported width explicitly.
3. **Inactive poison**: inactive values may contain NaN/Inf/arbitrary data and
   cannot affect logical forward results.
4. **All-empty**: `offsets[B] == 0` has defined behavior without reading inactive
   capacity.
5. **Reuse**: one placed executable handles shorter and longer logical extents
   without retaining stale runtime extent state.
6. **Persistence**: architecture/model save-load does not serialize payload-
   derived runtime partition caches; a loaded model accepts a different valid
   partition.
7. **Clone**: subgraph cloning remaps both values and structural offsets inputs
   when the operation participates in a cloned ragged graph.
8. **Backward**: when differentiable, gradients depend only on logical active
   values; an over-reading physical consumer owns sanitation of exactly the
   region it reads.
9. **Inference/training modes**: any mode-dependent behavior is tested in both
   paths where applicable.
10. **Capacity metadata**: operations that require `max_values_per_row` validate
    it at placement/runtime boundaries rather than inferring semantics from
    inactive values storage.

## Completion boundary

"Complete rank-1 ragged support" does not mean every dense layer accepts a
`RaggedTensor`. It means:

- operations with a clear sequence-ragged meaning have a first-class public
  contract;
- restricted operations fail explicitly outside that contract;
- partition-changing operations own the construction of their new offsets;
- no implementation relies on inactive packed capacity having canonical
  contents; and
- the support matrix and qualification gate remain synchronized with the
  shipped surface.
