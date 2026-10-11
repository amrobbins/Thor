# Ragged serialization ownership census

Status: `SERIALIZATION-OWNERSHIP-1A` census against the ThorKernels601 baseline.

This document records the current production artifact paths. It does **not** define or implement a new ragged schema.

## Executive finding

The current layer-local ragged JSON is not placement-independent.

`RaggedTensor::architectureJson()` emits placement geometry twice:

1. explicitly as `batch_size` and `max_total_values`; and
2. implicitly through ordinary `Tensor::architectureJson()`:
   - `values.dimensions[0] == max_total_values`;
   - `offsets.dimensions[0] == batch_size + 1`.

Removing only the explicit fields would therefore preserve the same placement geometry in the nested tensor shapes.

The RP7 network-boundary records are different. Current `ragged_network_inputs` v1.3 and
`ragged_network_outputs` v1.2 store logical tensor identities, shared-partition topology, and optional
`max_values_per_row`; they do not store `batch_size` or `max_total_values`.

A fully placement-independent layer-local ragged schema cannot be represented by the current logical
`Tensor`/`RaggedTensor` objects unchanged. `Tensor` has one concrete `dimensions` vector, while
`RaggedTensor(Tensor values, Tensor offsets, ...)` derives `maxTotalValues` from `values.dimensions[0]`
and `batchSize` from `offsets.dimensions[0] - 1`. The smallest blocker is therefore the logical tensor
shape contract at ragged edges, not the physical runtime connector implementation.

## Ownership classification

| Serialized fact | Ownership | Current emitted form | 1A behavior |
| --- | --- | --- | --- |
| Ragged tensor identity (`id`) | MODEL_RECONSTRUCTION | Layer-local `RaggedTensor` JSON | unchanged |
| Values tensor identity | MODEL_RECONSTRUCTION | nested `values.id`; RP7 boundary `values_tensor_id` | unchanged |
| Row-partition token identity | MODEL_RECONSTRUCTION | nested `offsets.id`; RP7 boundary `row_partition_token_tensor_id` | unchanged |
| Shared-partition source | MODEL_RECONSTRUCTION | RP7 input `partition_input_name` | unchanged |
| Values dtype and non-capacity/trailing dimensions | MODEL_RECONSTRUCTION | nested `Tensor` JSON | unchanged |
| Row-partition token dtype | MODEL_RECONSTRUCTION | nested `Tensor` JSON | unchanged |
| `max_values_per_row` when explicitly known | MODEL_RECONSTRUCTION | optional ragged/boundary field | unchanged |
| `batch_size` | RUNTIME_INSTANCE_STATE | explicit layer-local ragged field and implicit `offsets.dimensions[0] - 1` | unchanged legacy behavior |
| `max_total_values` | RUNTIME_INSTANCE_STATE | explicit layer-local ragged field and implicit `values.dimensions[0]` | unchanged legacy behavior |
| Legacy public offsets names/ids in pre-RP7 boundaries | COMPATIBILITY_ONLY | accepted by `Network::load` | unchanged |
| Learned parameters | LEARNED_MODEL_STATE | layer/parameter serialization, outside this census | unchanged |
| Optimizer/training progress | TRAINING_CHECKPOINT_STATE | checkpoint/optimizer serialization, outside this census | unchanged |
| Tensor-id substitutions performed by subgraph cloning | TRANSIENT_CLONE_JSON | temporary JSON in `Network::cloneSubgraphInto` | unchanged |

No field in this patch changes ownership or emitted production form.

## Core emitter/consumer census

### `RaggedTensor`

`DeepLearning/Api/Tensor/RaggedTensor.cpp`:

- `architectureJson()` emits version, ragged id/rank, `batch_size`, `max_total_values`, nested
  `values`, nested `offsets`, and optional `max_values_per_row`.
- `serialize(...)` emits the same schema, using `Tensor::serialize(...)` for the two nested tensors.
- `deserialize(...)` accepts versions 1.0.0 and 1.1.0, uses `Tensor::deserialize(...)` for both nested
  tensors, reconstructs a `RaggedTensor`, and validates the explicit batch/capacity against geometry
  derived from those tensors.

Repository-wide production call-site search finds no caller of `RaggedTensor::serialize(...)`.
Production model save goes through each layer's `serialize(...)`/architecture JSON and the network
boundary append path. `RaggedTensor::serialize(...)` is therefore currently a standalone API surface,
not a production model-artifact path.

Likewise, repository-wide production call-site search finds no caller of
`RaggedTensor::deserialize(...)`. Its direct callers are ragged tensor tests. Production layer and
network readers reconstruct ragged objects manually from tensors already known to the `Network`.

### Ordinary `Tensor`

`DeepLearning/Api/Tensor/Tensor.cpp` version 1.0.0 stores
`{version, id, dimensions, data_type}`. `Tensor::deserialize(...)` restores the concrete dimensions.

For a current logical `RaggedTensor` those concrete dimensions are placement-bearing:

- `values.dimensions[0]` is `max_total_values`;
- `offsets.dimensions[0]` is `batch_size + 1`.

This is the implicit placement channel that remains if only `RaggedTensor.batch_size` and
`RaggedTensor.max_total_values` are removed.

### Production layer families that emit nested ragged architecture JSON

The following production layer families directly call `RaggedTensor::architectureJson()` (or the
equivalent optional/vector member call) and therefore embed the current ragged schema:

- Activations: `Activation`.
- Learning: `Attention`, `Convolution1d`, `CustomLayer`, `Embedding`, `FullyConnected`,
  `ScaledDotProductAttention`.
- Metrics: `BinaryAccuracy`, `CategoricalAccuracy`, `ReductionMetrics`.
- Utility: `AdaptiveLayerNorm`, `Add`, `Concatenate`, `DropOut`, `FiniteCheck`, `Flatten`,
  `LayerNorm`, `PaddedDenseToRagged`, `RMSNorm`, `RaggedGather`, `RaggedRowLengths`,
  `RaggedSequenceConcatenate`, `RaggedSequenceSlice`, `RaggedToPaddedDense`, `Reshape`,
  `ScaleGradient`, `SegmentedBroadcast`, `SegmentedLogSoftmax`, `SegmentedReduction`,
  `SegmentedSoftmax`, `Slice`, `StopGradient`, `Transpose`, `TypeConverter`.
- Loss: `AsymmetricPowerLoss`, `BinaryCrossEntropy`, `BinaryFocalLoss`,
  `CategoricalCrossEntropy`, `CategoricalFocalLoss`, `CtcLoss`, `ExpectileLoss`,
  `GammaNLLLoss`, `GaussianNLLLoss`, `HuberLoss`, `LaplaceNLLLoss`,
  `MeanAbsoluteError`, `MeanAbsolutePercentageError`, `MeanPowerError`,
  `MeanSquaredError`, `NegativeBinomialNLLLoss`, `PoissonNLLLoss`, `QuantileLoss`,
  `RaggedCustomLoss`, `RaggedLossShaper`, `SmoothL1Loss`, `SoftTargetCrossEntropy`,
  `StudentTNLLLoss`, `TweedieLoss`.

`ReductionMetrics` has architecture-json code in both its header/template path and `.cpp`, but is one
layer family.

### Consumers using `SegmentedPrimitiveDetail`

`DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h` contains the shared
`reconstructInput(...)` helper. It resolves nested `values.id` and `offsets.id`, constructs a
`RaggedTensor`, then requires the reconstructed batch/capacity to equal serialized `batch_size` and
`max_total_values`.

The following production readers use that helper and therefore **require both explicit placement
fields indirectly**:

- Loss: `AsymmetricPowerLoss`, `BinaryCrossEntropy`, `BinaryFocalLoss`,
  `CategoricalCrossEntropy`, `CategoricalFocalLoss`, `ExpectileLoss`, `GammaNLLLoss`,
  `GaussianNLLLoss`, `HuberLoss`, `LaplaceNLLLoss`, `MeanAbsoluteError`,
  `MeanAbsolutePercentageError`, `MeanPowerError`, `MeanSquaredError`,
  `NegativeBinomialNLLLoss`, `PoissonNLLLoss`, `QuantileLoss`, `RaggedCustomLoss`,
  `RaggedLossShaper`, `SmoothL1Loss`, `SoftTargetCrossEntropy`, `StudentTNLLLoss`,
  `TweedieLoss`.
- Metrics: `BinaryAccuracy`, `CategoricalAccuracy`, `ReductionMetrics`.
- Utility: `PaddedDenseToRagged`, `RaggedGather`, `RaggedToPaddedDense`,
  `SegmentedBroadcast`, `SegmentedLogSoftmax`, `SegmentedSoftmax`.

The shared `validateSerializedPreservedPartition(...)` helper additionally validates output
`offsets.id`, `batch_size`, `max_total_values`, and optional `max_values_per_row`. It is used by:
`CategoricalCrossEntropy`, `RaggedCustomLoss`, `PaddedDenseToRagged`, `RaggedGather`,
`SegmentedBroadcast`, `SegmentedLogSoftmax`, and `SegmentedSoftmax`.

### Consumers with duplicated strict placement checks

These production readers implement equivalent local reconstruction/validation and directly read
`batch_size` and/or `max_total_values` instead of using the common helper:

- Activations: `Activation`.
- Learning: `Attention`, `Convolution1d`, `CustomLayer`, `Embedding`, `FullyConnected`.
- Loss: `CtcLoss`.
- Utility: `AdaptiveLayerNorm`, `DropOut`, `FiniteCheck`, `Flatten`, `LayerNorm`,
  `RMSNorm`, `RaggedRowLengths`, `RaggedSequenceConcatenate`, `RaggedSequenceSlice`,
  `Reshape`, `ScaleGradient`, `SegmentedReduction`, `Slice`, `StopGradient`, `Transpose`,
  `TypeConverter`.

These readers are migration sites for any future layer-local schema that removes explicit
batch/capacity.

### Manual readers that already ignore the explicit fields

`Add`, `Concatenate`, and `ScaledDotProductAttention` manually rebuild `RaggedTensor` values/partition
pairs without reading `batch_size` or `max_total_values`. They are **not** placement-independent:
construction still derives those values from the nested `Tensor` dimensions serialized in the same
ragged object.

This distinction is important: these three readers demonstrate that explicit fields are redundant
for reconstruction today, but not that the underlying artifact geometry is placement-free.

## RP7 network boundary metadata

`Network::appendRaggedNetworkBoundaryJson(...)` currently writes:

- `ragged_network_inputs` v1.3.0:
  - `name`;
  - `values_input_name`;
  - `values_tensor_id`;
  - owner only: `row_partition_token_tensor_id` and optional `max_values_per_row`;
  - shared partition only: `partition_input_name`.
- `ragged_network_outputs` v1.2.0:
  - `name`;
  - `values_output_name`;
  - `values_tensor_id`;
  - `row_partition_token_tensor_id`;
  - optional `max_values_per_row`.

Neither current boundary record contains `batch_size` or `max_total_values`.

For a shared input, the boundary record points at the canonical partition-owning logical input rather
than duplicating row-partition geometry. The shared record intentionally does not repeat
`row_partition_token_tensor_id` or `max_values_per_row`.

`Network::load(...)` accepts input boundary versions 1.0.0 through 1.3.0 and output boundary versions
1.0.0 through 1.2.0. Current RP7 input v1.3 reads `row_partition_token_tensor_id`; older input versions
read `offsets_tensor_id`. Current output v1.2 reads `row_partition_token_tensor_id`; older output
versions read `offsets_output_name` plus `offsets_tensor_id`.

The RP7 boundary record itself is model-oriented. However, its referenced API tensors were created
from layer JSON whose ordinary `Tensor.dimensions` still carry the old batch/capacity. The clean
boundary schema therefore does not make the complete model artifact placement-independent by itself.

## `RaggedNetworkInput::Builder`

`RaggedNetworkInput::Builder` currently requires concrete placement geometry for a partition-owning
input:

- `maxTotalValues(...)` creates the leading values dimension;
- `batchSize(...)` creates a hidden row-partition token with extent `batch + 1`;
- optional `maxValuesPerRow(...)` is semantic reconstruction information.

For a shared partition, the builder correctly inherits partition identity/geometry from the canonical
owner and rejects redundant batch/capacity arguments.

1A does not change this builder. Its use of concrete `Tensor` dimensions is another manifestation of
the same logical-shape blocker described above.

## Clone-only JSON rewriting

`Network::cloneSubgraphInto(...)` obtains layer architecture JSON, recursively rewrites API tensor ids
in that temporary JSON, and deserializes it into the destination graph. The helpers
`rewriteApiTensorIdsForClone(...)` and `rewriteAlreadyMappedApiTensorIdsForClone(...)` are therefore
`TRANSIENT_CLONE_JSON` behavior.

They are not save-time artifact migration and are not evidence for recursively rewriting completed
model JSON. 1A intentionally leaves both clone helpers unchanged.

## Versions and compatibility obligations

### Ragged tensor JSON

- Current writer: 1.1.0.
- Accepted reader versions: 1.0.0 and 1.1.0.
- v1.0.0 compatibility is characterized by tests as lacking `max_values_per_row`.
- Both legacy versions contain explicit `batch_size`/`max_total_values` and nested ordinary tensors
  whose dimensions independently encode the same geometry.

A future reader must continue to understand those legacy fields and nested shapes. It must not infer
semantic `max_values_per_row` from `max_total_values / batch_size`; absence means that bound was not
serialized.

### Network boundary JSON

- Ragged input current writer: 1.3.0; reader accepts 1.0.0-1.3.0.
- Ragged output current writer: 1.2.0; reader accepts 1.0.0-1.2.0.
- Pre-RP7 boundary ids/names remain compatibility-only inputs to `Network::load`.

No production version changes in 1A.

## Smallest target schema

The smallest useful future layer-local ragged model schema should contain only semantic graph
reconstruction facts:

- ragged identity/rank;
- values logical tensor identity, value dtype, and trailing (non-capacity) dimensions;
- row-partition logical identity and offsets dtype;
- optional explicit `max_values_per_row`;
- shared-partition relationships where relevant.

It should not contain:

- `batch_size`;
- `max_total_values`;
- a values leading dimension whose meaning is max capacity;
- a row-partition tensor extent whose meaning is batch + 1.

This is a target ownership contract, not a schema/version implemented by 1A.

## Smallest blocker and next boundary (1A finding)

The current logical API cannot instantiate that target schema unchanged because:

1. `Tensor` requires concrete dimensions at construction/deserialization.
2. `RaggedTensor` derives `maxTotalValues` and `batchSize` from those concrete tensor dimensions.
3. Network/layer reconstruction resolves tensor ids to those already-sized logical tensors before
   rebuilding `RaggedTensor`.
4. Placement later consumes the graph built from those logical objects.

Therefore the smallest concrete blocker is **the representation of ragged logical edge shape inside
ordinary `Tensor.dimensions`**. The evidence does not require a physical ragged connector overhaul.

At the conclusion of 1A, the next bounded sequence was to define and implement the smallest logical/API representation change
that lets ragged values preserve dtype/trailing shape and row-partition identity without fixing
capacity/batch in ordinary tensor dimensions, together with an explicit placement-time materialization
boundary. It should migrate one common reconstruction path first rather than rewriting every layer or
completed artifact JSON at once.

That next sequence must preserve legacy v1.0/v1.1 reading and must not manufacture
`max_values_per_row` from historical batch/capacity.

## 1B placement-neutral external boundary

`SERIALIZATION-OWNERSHIP-1B` adds a deliberately narrow second geometry mode to the existing API
`RaggedTensor` for external network boundaries. Legacy/fixed ragged tensors retain their concrete
`[max_total_values, ...]` values shape and `[batch_size + 1]` partition-token shape. A placement-neutral
boundary instead uses `[max_values_per_row, ...]` for the logical values tensor and canonical `[1]` for
the graph-only row-partition token. The mode is explicit; no geometry is inferred by dividing legacy
batch/capacity fields.

Neutral `ragged_network_inputs` use boundary version 1.4.0 and neutral `ragged_network_outputs` use
version 1.3.0, both with `geometry_mode = "placement_neutral"`. The owner record carries explicit
`max_values_per_row`; a shared input carries only its canonical `partition_input_name` and inherits the
owner partition semantics. Legacy input v1.0-v1.3 and output v1.0-v1.2 records remain fixed-geometry
compatibility paths.

At placement, `RaggedTensor::getDescriptorForPlacement(B)` derives physical packed capacity `B * M`
and physical offsets extent `B + 1`, rejecting multiplication overflow and capacities that the offsets
dtype cannot represent. The logical API tensors are not mutated. `Network` uses that descriptor only
for the existing input stamping, row-partition physicalization, binding, and placement memory-accounting
paths; `DatasetInputBindings` likewise derives its concrete residency requirement from the dataset
session batch. Generic layer-local `RaggedTensor::architectureJson()` remains fixed-geometry-only, so
the 58 layer-local ragged schemas catalogued above are intentionally still migration work for the next
sequence.
