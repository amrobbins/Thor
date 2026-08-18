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

The canonical offsets tensor is the semantic row partition. For batch size `B`, `offsets[B]` is the active packed-value count and the exclusive logical end of the packed values tensor.

Packed capacity in `[offsets[B], maxTotalValues)` is undefined storage. It is not padding with a semantic value, and neither internal layers nor callers outside a network may rely on it being zero, finite, stable, or otherwise canonical. The same rule applies at `RaggedNetworkInput` and `RaggedNetworkOutput` boundaries.

Thor uses a consumer-responsibility policy for physical kernels. An active-aware consumer executes only the logical extent and ignores inactive capacity. If a physical implementation deliberately chooses an execution extent larger than `offsets[B]` (for example, a bucketed GEMM), that consumer must sanitize exactly the additional region it will read immediately before the read. After the consumer finishes, that region is undefined again. Producers do not canonicalize inactive capacity merely because they produced a ragged tensor.

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

## Network boundaries and inactive capacity

A logical `RaggedNetworkInput` materializes values and offsets through separate physical input ports. The submitted host-known active count is boundary execution metadata used to publish the offsets-owned runtime cache after the new offsets payload is materialized. It does not extend the logical tensor into inactive packed capacity.

The consumer-responsibility contract continues unchanged across both network boundaries:

- `RaggedNetworkInput` does not semantically promise zero or canonical values in `[offsets[B], maxTotalValues)`;
- `RaggedNetworkOutput` exposes the values capacity and the authoritative offsets without assigning semantics to the inactive tail;
- code consuming a ragged output after it leaves a network must treat `offsets[B]` as the logical boundary exactly as an internal consumer would.

Current implementations may temporarily perform broader sanitation while the consumer-responsibility cleanup is staged. That behavior is incidental and is deliberately not part of the public or internal semantic contract. Tests therefore compare logical prefixes and poison inactive capacity rather than asserting tail bytes.

The legacy flattened `map<string, Tensor>` submission surface cannot represent the logical ragged boundary contract and is rejected for stamped networks with `RaggedNetworkInput`s. Submit a `Batch` containing `RaggedTensor` entries instead.

## Expression execution

Packed Expression operations retain the canonical offsets tensor as a structural stage input when they need runtime extent. Active-aware valuewise stages execute only the logical prefix. Bucketed physical operations such as MATMUL (and, after the RMSNorm lifecycle cleanup, bucketed RMSNorm) may choose a larger execution extent; each such consumer owns sanitation of exactly the bucket slack it will physically read. Expression values do not carry a parallel active-row annotation.

## Regression gates

The row-partition runtime tests compile-time check that implementation `Tensor` has no legacy active-row getter, setter, or clearer. End-to-end ragged transformer tests exercise FullyConnected, activations, RMSNorm, DropOut, self-attention, both mixed dense/ragged attention quadrants, residual Add/CustomLayer, segmented reduction, backward training, inference, save/load, and changing packed extents.

Save/load coverage deliberately saves after one active packed count and reloads with a different count, then reuses one placed model across shorter and longer partitions. This guards against accidentally serializing or retaining the ephemeral host cache.

## Audit coverage

The cutover is covered at several levels rather than by one metadata-propagation test:

| Surface | Primary regression coverage |
| --- | --- |
| Runtime ownership and generic copies | `RowPartitionRuntime.*`, `RaggedTensorImplementation.*` |
| Logical ragged input/output boundary semantics | `RaggedNetworkOutputApi.*`, Python `test_placed_network.py` |
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
