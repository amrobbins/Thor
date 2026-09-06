# Rank-1 ragged support contract

This document is the repository-level contract and support matrix for Thor's
first-class ragged execution surface. It describes what is supported today,
which restrictions are intentional, and which dense operations are not yet
part of the ragged API.

R11D is the closure/qualification point for the initial rank-1 ragged support
campaign. The matrix below describes the shipped surface after R11A-R11C.1 and
the RP1-RP7 row-partition runtime migration. New ragged capabilities must keep
this matrix and the `check-ragged-support-contract` gate synchronized.

## Scope

Thor's public `RaggedTensor` is a **canonical rank-1 sequence-ragged tensor**:

```text
RaggedTensor
  values:        [max_total_values, trailing...]
  row partition: logical RowPartitionId + authoritative host offsets
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
- the complete host offsets vector is the authoritative runtime row partition;
  `active_value_count` and `max_active_row_length` are derived from it and cannot
  be published independently;
- the graph-local partition token is topology only and does not imply a device
  offsets allocation;
- physical partition representations are requirement-driven: a consumer may
  request host extent metadata, a managed `[1]` device active-count scalar, or
  managed `[B+1]` device offsets;
- runtime host partition state is never serialized as model state.

Nested ragged ranks are outside this contract. A future nested/jagged tensor
abstraction must not silently change the meaning of the existing rank-1 type.

The physical ownership and consumer-responsibility rules are specified in
[`ragged_row_partition_runtime.md`](ragged_row_partition_runtime.md).

## Operation classes

A **partition-preserving** operation transforms values but retains the exact
authoritative row partition:

```text
(values A, partition P) -> operation -> (values B, partition P)
```

A **partition-changing** operation changes row membership or segmentation and
must explicitly produce a new authoritative host partition (plus any required
execution-side offsets representation):

```text
(values A, partition P) -> repartition -> (values B, partition Q)
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
| `RaggedNetworkInput` / `RaggedNetworkOutput` | Supported | Every executing logical boundary has complete authoritative host offsets. Fresh public graph boundaries expose values only; the graph-local partition token is hidden topology and is never stamped or dataset-bound. CPU offsets supplied in a logical batch establish the host partition directly; GPU-only offsets without host partition state are rejected. `partition=<existing ragged input>` declares a values-only stream sharing that exact partition. Ragged outputs reconstruct their returned offsets from authoritative host state without adding a physical offsets output or D2H synchronization; inactive capacity remains undefined | `RaggedNetworkInputApi.*`, `RaggedNetworkOutputApi.*` |
| `RaggedTensor` serialization / descriptors | Supported | `UINT32` and `UINT64`; optional `max_values_per_row`; authoritative runtime host partition state is not serialized | `RaggedTensorApi.*`, implementation descriptor/runtime tests |
| `NumpyDataset` host/device residency | Supported | Canonical packed values + offsets materialize through both host-backed batching and generic device-resident snapshots; STRICT device storage supports dense/ragged mixtures and enforces requested `max_total_values` / `max_values_per_row` bounds | Python `test_numpy_dataset_ragged_batches_train_through_canonical_ctc_with_exact_partial_tail`, device-resident named-session capacity tests |
| Ordinary standalone activations | Supported | Tokenwise over active packed values and partition-preserving | `Activations.ShapePreservingBuildersInferRaggedFromInputAndPreserveOffsets` |
| GLU-family activations | Supported | Tokenwise; final trailing feature dimension must satisfy the gate/split geometry | `GatedLinearUnits.Ragged*` |
| `Softmax` | Supported | Ordinary tokenwise softmax over the final trailing/channel dimension for every active packed value; exact partition preservation; distinct from sequence-axis `SegmentedSoftmax`; FP16/BF16/FP32 values only | `Activations.RaggedSoftmax*`, Python `test_ragged_softmax_*`, `RaggedExpression.OrdinarySoftmax*` |
| `SegmentedSoftmax` / `SegmentedLogSoftmax` | Supported | Normalize independently across each variable-length row for every trailing component; exact partition preservation; FP16/BF16/FP32 values only, with FP64 intentionally unsupported | `SegmentedPrimitiveApi.*`, Python `test_segmented_primitives.py`, `RaggedExpression.*SegmentSoftmax*` |
| `FullyConnected` | Restricted | Tokenwise final-dimension projection; prefix preservation required; ordinary supported activation contract applies; custom epilogues are supported; auxiliary bindings must be same-partition `RaggedTensor` inputs and are active-prefix-aware | `FullyConnectedApi.Ragged*`, Python `test_fully_connected.py` |
| `RMSNorm` | Restricted | Normalization axes must stay within trailing value dimensions; custom epilogues are supported; auxiliary bindings must be same-partition `RaggedTensor` inputs and remain active-prefix-aware | `UtilityApiLayers.RaggedRMSNorm*`, Python `test_rms_norm.py` |
| `LayerNorm` | Restricted | Public ragged LayerNorm is token-wise over exactly one non-zero trailing channel dimension; `normalizedShape` must equal that trailing dimension. Multi-axis ragged normalization remains outside the current expression backend contract | `UtilityApiLayers.RaggedLayerNorm*`, Python `test_layer_norm.py`, `RaggedExpression.*LayerNorm*` |
| `DropOut` | Supported | Training touches only the logical extent; inference/validation identity preserves the partition | `DropOut.Ragged*`, `UtilityApiLayers.RaggedDropOut*` |
| `Add` | Restricted | Both operands must share the exact same logical row partition and have compatible values descriptors | ragged `CustomLayer`/transformer integration coverage |
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
| ordinary valuewise losses | Restricted | `MAE`, `MSE`, `MeanPowerError`, `MAPE`, `HuberLoss`, `SmoothL1Loss`, `QuantileLoss`, `ExpectileLoss`, `AsymmetricPowerLoss`, `BinaryCrossEntropy`, and `BinaryFocalLoss` accept same-partition ragged predictions/labels and retain the dense dtype/math contract. Dense `[1]` per-example weights are supported only where the dense loss already exposes weights and are broadcast over the corresponding logical row. `RAW` preserves the prediction partition, `PER_EXAMPLE` sums active token/output contributions per logical row, `BATCH` averages row sums over valid logical examples, `NONE` remains a training-only root, and `PER_OUTPUT` is rejected. MAPE retains its dense stability clamp. | `RaggedMAEApi.*`, `RaggedRegressionR10E.*`, `RaggedRegressionR10F.*`, `RaggedRegressionR10G.*`, `RaggedClassificationR10H.*`; common execution/shaping in `RaggedCustomLoss.*` / `RaggedLossShaper.*` |
| distribution losses | Restricted | `PoissonNLLLoss`, `TweedieLoss`, `GammaNLLLoss`, `GaussianNLLLoss`, `LaplaceNLLLoss`, `StudentTNLLLoss`, and `NegativeBinomialNLLLoss` support rank-1 ragged tokenwise likelihoods. Every token-varying differentiable parameter must share the exact prediction/label partition and compatible value geometry; fixed scalar parameters retain their dense contracts. Reporting uses the same ragged `RAW`/`PER_EXAMPLE`/`BATCH`/`NONE` policy and rejects `PER_OUTPUT`. | `RaggedDistributionR10I.*`, `RaggedDistributionR10J.*`, `RaggedDistributionR10K.*`, Python distribution-loss tests |
| categorical losses | Restricted | `CategoricalCrossEntropy`, `SoftTargetCrossEntropy`, and `CategoricalFocalLoss` support dense-target ragged logits with exactly one trailing class dimension `[C]`, normalize independently across that class axis for every active token, and preserve the exact row partition for `RAW`. `SparseCategoricalCrossEntropy` supports same-partition scalar integer class ids (`[]` or `[1]`) plus an optional same-partition scalar mask and uses a logits-native active-prefix kernel. Sparse `RAW`/`NONE` need only the active-count representation; row-reduced reporting materializes full offsets only because the shaper genuinely needs row boundaries. `PER_OUTPUT` is rejected for every ragged categorical loss. | `RaggedCategoricalR11C.*`, `RaggedSparseCategoricalR11C1.*`, `SparseCategoricalCrossEntropyWithLogits.Ragged*`, Python categorical-loss tests |
| reduction metrics | Restricted | `Sum`, `Mean`, `Min`, `Max`, and `WeightedMean` accept ragged values. `Sum` aggregates active scalars by sum; `Mean` and `WeightedMean` publish numerator/denominator sufficient statistics for exact epoch-wide ratio aggregation; `Min`/`Max` ignore empty rows and publish explicit no-contribution metadata for all-empty batches. `WeightedMean` requires same-partition ragged values/weights. FP8 E4M3/E5M2, FP16, BF16, and FP32 value storage are supported; FP64 and integer storage are intentionally rejected. | `ReductionMetricApi.R10L*`, `ReductionMetricApi.R10M*`, `ReductionMetricApi.R10N*`, `RaggedWeightedReduction.*`, `MetricEpochAccumulator*.*`, Python reduction-metric tests |
| accuracy metrics | Restricted | `BinaryAccuracy` and `CategoricalAccuracy` operate over active tokens of same-partition ragged predictions/labels and aggregate exact correct-count/token-count sufficient statistics across partial and unequal batches. Categorical accuracy supports per-class labels and integer class-index labels with the documented class-axis geometry. Ragged prediction storage is FP16/FP32. | `RaggedAccuracyApi.*`, `RaggedAccuracy.*`, `MetricEpochAccumulator.R10O*`, Python binary/categorical accuracy tests |
| `CustomMetric` / `LossMetric` ragged inputs | Not public | These generic metric builders remain dense-input surfaces. First-class ragged metric coverage is provided by the reduction and accuracy metric classes above; adding generic ragged expression-backed metrics requires a separate contract for partition-aware metric outputs/statistics. | dense metric tests |
| `Embedding` | Supported | Integer ragged indices map tokenwise to floating embeddings while preserving the exact logical partition; forward and sparse backward are bounded by the authoritative active-value count, never inspect inactive packed index/gradient capacity, reuse captured sparse-update graphs across changing active extents, and support FP16/BF16/FP32 embedding values with UINT8/16/32/64 indices and UINT32/64 partition offsets | `EmbeddingApi.Ragged*`, `EmbeddingRaggedRuntimeTest.*`, Python `test_embedding.py` |
| `FiniteCheck` | Supported | Zero-copy diagnostic identity; forward/backward inspect only the authoritative active packed prefix using the managed active-count representation; inactive capacity remains undefined and is ignored | `UtilityApiLayers.RaggedFiniteCheck*`, `FiniteCheck.Ragged*`, Python `test_finite_check.py` |
| `StopGradient` / `ScaleGradient` | Supported | Forward preserves the exact logical row partition; gradient control applies only to packed values and never changes row membership | `UtilityApiLayers.RaggedStopGradient*`, `UtilityApiLayers.RaggedScaleGradient*`, Python gradient-control tests |
| trailing `Reshape` / `Flatten` | Supported | Metadata-only transforms of each packed value's trailing shape; element count per packed value and the exact canonical row partition are preserved | `UtilityApiLayers.RaggedReshape*`, `UtilityApiLayers.RaggedFlatten*`, Python `test_ragged_shape_ops.py` |
| trailing `Transpose` | Restricted | Swaps only the final two trailing value dimensions; the packed row axis is never transposed. Materialization/backward uses an active-prefix non-overlapping strided view | `UtilityApiLayers.RaggedTranspose*`, `RaggedExpression.TrailingTranspose*`, Python `test_ragged_shape_ops.py` |
| `RaggedSequenceConcatenate` | Supported | Concatenates corresponding rows in input order across independently partitioned rank-1 ragged inputs; same batch/values dtype/offsets dtype/trailing shape required; derives authoritative host partition `Q` from source host offsets before GPU value movement. The concatenate value kernel consumes source partitions; physical representations of `Q` are materialized only when downstream requirements request them. Forward/backward touch only active packed values | `UtilityApiLayers.RaggedSequenceConcatenate*`, `RaggedSequenceConcatenate.*`, Python `test_ragged_sequence_concatenate.py` |
| `RaggedSequenceSlice` | Supported | Applies a fixed non-negative `start` and positive `length` independently to each logical row, clips short rows, and derives authoritative output partition `Q` on the host before GPU value movement. Slice itself genuinely needs row starts for compaction, so `Q`'s managed `[B+1]` representation is a physical input to the values-only Slice implementation; downstream representations remain requirement-driven. Backward zeros sliced-out active source gradients and scatters selected gradients without touching inactive capacity | `UtilityApiLayers.RaggedSequenceSlice*`, `RaggedSequenceSlice.*`, Python `test_ragged_sequence_slice.py` |
| `RaggedGather` | Supported | Interprets scalar UINT32/UINT64 indices row-locally against source partition P; output values use source dtype/trailing geometry and reuse the indices partition Q exactly; source and indices offsets dtypes may differ; duplicate indices accumulate during backward; inactive capacities are ignored | `UtilityApiLayers.RaggedGather*`, `RaggedGather.*`, Python `test_ragged_gather.py` |
| `RaggedFilter` / runtime device-data-dependent repartitioning | Out of scope | Thor requires every live ragged partition to have authoritative host offsets before execution. A filter whose output row lengths depend on runtime device BOOLEAN values cannot satisfy that contract without a device-to-host synchronization, so the former `RaggedFilter` layer was removed. Perform such filtering before batch submission, where the resulting host partition can be constructed directly. | none |
| `RaggedToPaddedDense` | Supported | Losslessly materializes canonical ragged rows as ordinary dense `[B,W,...]` storage using the declared finite `max_values_per_row=W`; inactive packed capacity is ignored, short rows are filled with an explicit constant padding value, and backward discards padding gradients | `UtilityApiLayers.RaggedDenseAdapters*`, `RaggedDenseAdapters.*`, Python `test_ragged_dense_adapters.py` |
| `PaddedDenseToRagged` | Supported | Packs ordinary dense `[B,W,...]` storage according to an existing canonical `partition_input`; only that partition's row boundaries are consumed, `W` must cover `max_values_per_row`, output reuses the exact logical partition, padding cells are ignored, and backward emits exact-zero padded gradients | `UtilityApiLayers.RaggedDenseAdapters*`, `RaggedDenseAdapters.*`, Python `test_ragged_dense_adapters.py` |
| `AdaptiveLayerNorm` | Restricted | Rank-1 `[N,C]` ragged data uses dense per-logical-row `[C]` scale/bias inputs; `SegmentedBroadcast` expands each row's conditioning only across its active tokens, the exact partition is preserved, and normalization reuses packed finite-bucket `LayerNorm`. Multi-axis ragged normalization remains out of scope and training inherits the current packed `LayerNorm` autodiff gate | `UtilityApiLayers.RaggedAdaptiveLayerNorm*`, Python `test_ragged_adaptive_layer_norm_*` |
| `BatchNorm`, `InstanceNorm`, Conv2D/3D, 2D pooling, unrestricted `Einsum` | Out of scope | A single unambiguous rank-1 sequence-ragged meaning has not been standardized | dense-only public surfaces |

## Qualification gate

The repository-level C++ closure gate is:

```bash
cmake --build <build-dir> --target check-ragged-support-contract
```

R11D deliberately gathers existing numerical/API tests instead of duplicating
those tests in a monolithic suite. The target runs a disabled CUDA preflight so
a GPU-less machine cannot pass merely because all CUDA-backed constituents
skipped. The heavier retained ragged Conv1D performance/timing qualification
remains separately enforced by `check-retained-ragged-training-production-gate`.

The final gate covers the following closure requirements:

| Required behavior | Gathered regression evidence |
| --- | --- |
| Both `UINT32`/`UINT64` offset widths; canonical boundaries; inactive poison; interleaved/empty/all-empty rows; short -> long -> short reuse | `RaggedSupportContract.*`, layer-specific ragged suites, `RaggedSparseCategoricalR11C1.*`, metric R10L-R10O suites |
| Save/load with a changed runtime partition; graph/subgraph clone remapping | `RaggedSupportContract.*`, `RaggedTransformerCompleteness.*`, `TrainingRunsResult.Save*Ragged*`, `RaggedSparseCategoricalR11C1.*` |
| Forward/backward active-prefix safety and weighted losses | `RaggedMAEApi.*`, R10E-R10K loss suites, `RaggedCategoricalR11C.*`, `RaggedSparseCategoricalR11C1.*`, `RaggedCustomLoss.*`, `RaggedLossShaper.*` |
| Ordinary final-axis Softmax differs from segmented row-axis Softmax/LogSoftmax; FP16/BF16/FP32 qualified and FP64 intentionally rejected | `Activations.RaggedSoftmax*`, `CudnnRaggedSoftmaxDescriptor.*`, `CudnnRaggedSoftmaxR11A1.*`, `RaggedExpression.OrdinarySoftmax*`, `RaggedExpression.*Segment*Softmax*` |
| Every public ragged metric class; partial batches; weighted mean; exact epoch-wide ratio aggregation; no-contribution extrema/zero-weight batches | `ReductionMetricApi.R10L*`, `ReductionMetricApi.R10M*`, `ReductionMetricApi.R10N*`, `RaggedAccuracyApi.*`, `RaggedAccuracy.*`, `RaggedWeightedReduction.*`, `MetricEpochAccumulator*.*` |
| Representative structural/learning/attention/data-adapter surfaces | FullyConnected/RMSNorm/LayerNorm/DropOut/Embedding/FiniteCheck, segmented primitives, shape ops, sequence concatenate/slice/gather, ragged-dense adapters, attention, CTC, and public causal Conv1D suites listed in the CMake filter |

Python API parity should be checked with the corresponding focused suites. The
Python data tests additionally qualify host-backed and STRICT device-resident
ragged `NumpyDataset` materialization. A useful closure run is:

```bash
pytest -q \
  bindings/python/test/core/activations/test_ragged_activations.py \
  bindings/python/test/core/losses \
  bindings/python/test/core/metrics \
  -k 'ragged or r10 or r11 or softmax'
```

The Python suite is not invoked by the CMake C++ gate because a configured
Python test environment is not a prerequisite for building the C++ library.

## Required regression pattern for new ragged operations

A new first-class ragged operation should not be marked **Supported** until its
tests address the applicable parts of this checklist:

1. **Structure**: the exact logical `RowPartitionId` is preserved for a
   partition-preserving operation, or a new authoritative host partition is
   explicitly derived/adopted for a partition-changing operation.
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
7. **Clone**: subgraph cloning remaps both values topology and the graph-local
   partition token/`RowPartitionId` when the operation participates in a cloned
   ragged graph.
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
- partition-changing operations own construction of their new authoritative host partition;
- no implementation relies on inactive packed capacity having canonical
  contents; and
- the support matrix and qualification gate remain synchronized with the
  shipped surface.
