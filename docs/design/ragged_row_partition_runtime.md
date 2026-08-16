# Ragged row-partition runtime model

Thor's native ragged representation separates packed values from row-partition state:

```text
RaggedTensor
  values: Tensor
  rowPartition: RowPartitionRuntime
    offsets: Tensor
    descriptor: RowPartitionDescriptor
```

The values tensor is an ordinary dense-capacity tensor. It carries no ragged runtime metadata.

## Source of truth

The canonical offsets tensor is the semantic row partition. For batch size `B`, `offsets[B]` is the active packed-value count.

`RowPartitionRuntime` may cache that terminal offset on the host. The cache exists only to support operations that must make a host-side dispatch decision, such as selecting a pre-tuned packed GEMM capacity bucket. It is not a second semantic representation of the partition.

CPU offsets can provide the terminal value directly when no explicit cache is present. GPU offsets require the cache for host-dispatched operations.

## Lifetime and persistence

The host active-value count belongs to the backing allocation of the canonical offsets tensor. Independent `RowPartitionRuntime` wrappers around the same canonical offsets allocation therefore share one cache.

The cache is ephemeral execution state and is payload-derived:

- generic Tensor mutation invalidates any cache attached to the destination allocation;
- generic `Tensor::copyFromAsync()` never propagates the source cache;
- a newly allocated offsets tensor starts with no cache;
- logical ragged input materialization copies offsets first, then republishes or clears the placed offsets cache before downstream execution;
- CPU cache publication and consumption verify that the cached count still equals `offsets[B]`;
- architecture and model/state serialization do not contain the cache.

Mutable raw CPU access can bypass Tensor mutation hooks. `RowPartitionRuntime` therefore rechecks the terminal CPU offset whenever an explicit cache is consumed and fails on disagreement instead of returning stale structural state. GPU code that directly materializes canonical offsets is responsible for publishing or clearing the cache at that structural boundary.

A freshly loaded and placed model obtains runtime state only from newly submitted row partitions.

## Partition-preserving operations

A partition-preserving operation changes packed values while retaining the exact row partition:

```text
(values A, partition P)
        |
        v
      layer
        |
        v
(values B, partition P)
```

FullyConnected, RMSNorm, training DropOut, Attention with ragged query/output, activations, TypeConverter, Slice, and ragged CustomLayer operations follow this model. Identity DropOut may alias the values tensor directly; the logical output still shares partition `P`.

Operations that need host dispatch receive or retain the offsets tensor as an explicit structural input and query `RowPartitionRuntime`. They never annotate their values output or input gradient with partition state.

## Partition-changing operations

An operation that changes row membership or segmentation must explicitly produce a new canonical offsets tensor and therefore a new row partition:

```text
(values A, partition P)
        |
        v
 segmented/repartitioning operation
        |
        v
(values B, partition Q)
```

There is no implicit propagation convention on values tensors that can create `Q` accidentally.

## Input-boundary tail canonicalization

A logical `RaggedNetworkInput` materializes values and offsets through separate physical input ports. The submitted host-known active count is carried as boundary metadata. Values use that count to canonicalize inactive capacity, while the offsets input publishes the count onto its canonical placed allocation only after the new offsets payload has been materialized and before downstream layers run.

This ordering lets generic Tensor writes invalidate stale payload-derived metadata without racing the ragged input boundary. It keeps inactive packed capacity deterministic while preserving zero-copy identity layers such as inference DropOut.

The legacy flattened `map<string, Tensor>` submission surface cannot represent this logical contract and is rejected for stamped networks with `RaggedNetworkInput`s. Submit a `Batch` containing `RaggedTensor` entries instead.

## Expression execution

Packed Expression MATMUL and RMSNorm retain the canonical offsets tensor as a structural stage input. Forward and autodiff stages query the corresponding `RowPartitionRuntime` for host bucket selection. Expression values do not carry a parallel active-row annotation.

## Regression gates

The row-partition runtime tests compile-time check that implementation `Tensor` has no legacy active-row getter, setter, or clearer. End-to-end ragged transformer tests exercise FullyConnected, activations, RMSNorm, DropOut, self-attention, both mixed dense/ragged attention quadrants, residual Add/CustomLayer, segmented reduction, backward training, inference, save/load, and changing packed extents.

Save/load coverage deliberately saves after one active packed count and reloads with a different count, then reuses one placed model across shorter and longer partitions. This guards against accidentally serializing or retaining the ephemeral host cache.

## Audit coverage

The cutover is covered at several levels rather than by one metadata-propagation test:

| Surface | Primary regression coverage |
| --- | --- |
| Runtime ownership and generic copies | `RowPartitionRuntime.*`, `RaggedTensorImplementation.*` |
| Logical ragged input/output and tail canonicalization | `RaggedNetworkOutputApi.*`, Python `test_placed_network.py` |
| Indexed/File/NamedBatch session production | `IndexedNamedBatchSessionTest.*` and device-resident ragged session tests |
| FullyConnected / packed MATMUL / autodiff | `FullyConnectedApi.Ragged*`, `RaggedExpression.*` |
| RMSNorm / packed RMSNorm / parameter gradients | `UtilityApiLayers.RaggedRMSNorm*`, `RaggedExpression.*` |
| DropOut training/inference identity behavior | `DropOut.Ragged*`, Python `test_drop_out.py` |
| Attention and mixed dense/ragged quadrants | `AttentionApi.*Ragged*`, Python `test_attention.py`, transformer tests |
| Ragged CustomLayer / residual Add | `RaggedCustomLayer.*`, Python `test_custom_layer_ragged.py` and transformer tests |
| TypeConverter and activations | Python `test_type_converter.py` and `test_ragged_activations.py` |
| Segmented reductions / ragged Slice | Python `test_segmented_reduction.py` |
| Architecture/model save-load | `RaggedTensorApi.*` and ragged transformer completeness test |
| TrainingPhase and Python training | ragged transformer TrainingPhase round trip and training integration test |

The final repository-level cutover gate is that the removed values-owned active-row identifier has no contiguous occurrences anywhere in the tree.
