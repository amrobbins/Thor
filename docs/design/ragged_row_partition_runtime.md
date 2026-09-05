# Ragged row-partition runtime model

The repository-level public support matrix and qualification checklist live in
[`ragged_support_contract.md`](ragged_support_contract.md). This document focuses
on physical row-partition ownership and runtime extent semantics.

Thor's native ragged representation separates packed values from row-partition state:

```text
RaggedTensor
  values: Tensor
  rowPartition:
    identity: RowPartitionId
    hostOffsets: authoritative runtime partition
    descriptor: RowPartitionDescriptor
    offsets: Tensor  # transitional execution representation
```

The values tensor is an ordinary dense-capacity tensor. It carries no ragged runtime metadata.

## Source of truth

The complete host offsets vector is the authoritative semantic row partition. For
batch size `B`, `hostOffsets[B]` is the active packed-value count and each pair
`hostOffsets[i:i+2]` defines one logical row. `activeValueCount` and
`maxActiveRowLength` are derived from that single publication and cannot be
updated independently.

Logical partition identity is distinct from the offsets `Tensor`. RP4 gives every
ragged value a stable row-partition identity and requires semantic compatibility checks
to use that identity rather than Tensor equality. While the offsets representation is
still mandatory during this transitional stage, new partition identities are seeded
from the existing live symbolic offsets identity so legacy construction and loaded
partition topology remain stable.

The offsets `Tensor` is an execution representation of the same partition, not a
second source of truth and not the definition of partition identity. Later migration
steps may allocate or materialize device offsets only for physical consumers that
actually need row boundaries on device. Generic Tensor payload mutation does not
redefine or invalidate the authoritative host partition.

Row partitions do not carry revision or generation counters. Device partition
representations are not hidden mutable caches whose freshness is inferred from
version state. If a GPU operation requires row boundaries or another partition
quantity, that representation must appear explicitly among the operation's physical
execution inputs and participate in ordinary stream/event dependency tracking.

Packed capacity in `[hostOffsets[B], maxTotalValues)` is undefined storage. It is
not padding with a semantic value, and neither internal layers nor callers outside a
network may rely on it being zero, finite, stable, or otherwise canonical. The same
rule applies at `RaggedNetworkInput` and `RaggedNetworkOutput` boundaries.

Thor uses a consumer-responsibility policy for physical kernels. An active-aware
consumer executes only the logical extent and ignores inactive capacity. If a
physical implementation deliberately chooses a larger execution extent (for example,
a bucketed GEMM), that consumer must sanitize exactly the additional region it will
read immediately before the read. After the consumer finishes, that region is
undefined again. Producers do not canonicalize inactive capacity merely because they
produced a ragged tensor.

## Lifetime and persistence

The logical row-partition identity belongs to the ragged partition, not to a device
offsets tensor. Partition-preserving operations propagate that identity; structural
partition-changing operations create or adopt a different identity. The current RP4
implementation still stores the authoritative host publication on the offsets backing
allocation as a transitional runtime-storage detail, so independently-created
`RowPartitionRuntime` wrappers around that same execution tensor share host state.
That storage coupling is not semantic identity and will disappear when device
partition inputs are physicalized explicitly in later patches.

A newly allocated runtime may be temporarily *unbound* during graph placement, before
a concrete batch partition is published. An executing supported ragged value must be
bound: every partition-owning network submission publishes complete host offsets, and
internal structural consumers require authoritative input offsets before executing.
Once `setHostOffsets()` is called, generic writes or copies of the offsets Tensor do
not change that logical partition. Only another complete host-offset publication can
change it. Supported execution paths do not clear the authoritative host
partition once a batch is bound.

The host partition is runtime execution state and is not serialized into architecture
or model/state files. Logical row-partition identity is graph-local as well: save/load
reconstructs a new identity from the loaded graph topology rather than persisting a
process-local handle. A freshly loaded and placed model obtains a new authoritative
host partition from subsequent batch submission.

RP1 removes the old independent host scalar setters/clearers entirely. This prevents
representing combinations such as an active count or maximum row length that disagree
with the row boundaries.

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

Semantic partition compatibility is checked with `sharesPartitionWith()`, never by
comparing offsets tensors. Operations that still need host dispatch may retain the
transitional offsets tensor as an explicit structural input and query
`RowPartitionRuntime`. They never annotate their values output or input gradient with
hidden partition state.

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

`RaggedSequenceConcatenate`, `RaggedSequenceSlice`, and `RaggedGather` are the supported examples of this family. Concatenate and slice derive complete output host offsets from authoritative input host offsets before GPU value movement, then materialize those host-derived offsets as an explicit GPU input where their kernels need row boundaries. Gather adopts the authoritative partition already owned by its ragged indices input. Dense-to-ragged conversion likewise requires an existing authoritative partition; the former low-level GPU-lengths-to-ragged path was removed. Thor does not support partition-producing operations whose output row boundaries depend on runtime device values, because establishing their authoritative host partition would require a device-to-host synchronization.

There is no implicit propagation convention on values tensors that can create `Q` accidentally.

## Network boundaries and inactive capacity

A partition-owning logical `RaggedNetworkInput` currently materializes values and the offsets execution representation through separate physical input ports. A shared-partition `RaggedNetworkInput(..., partition=source)` materializes only a new values port and reuses the same row-partition state. Every partition-owning batch submission must publish complete authoritative host offsets; active count and maximum row length are derived from that publication. CPU offsets supplied at the logical boundary may establish the host partition directly because they are already host-resident. GPU offsets without an accompanying authoritative host partition are rejected rather than synchronized back to the CPU.

At direct submission time, a shared-partition logical input may be supplied as packed values only. The referenced partition-owning input supplies the row partition for the batch. Supplying a full ragged value for a shared input remains accepted for dataset/materialization compatibility; its offsets are not wired as a second graph boundary, and its authoritative host offsets must agree with the owner's partition.

The consumer-responsibility contract continues unchanged across both network boundaries:

- `RaggedNetworkInput` does not semantically promise zero or canonical values in `[offsets[B], maxTotalValues)`;
- `RaggedNetworkOutput` exposes the values capacity and the authoritative offsets without assigning semantics to the inactive tail;
- code consuming a ragged output after it leaves a network must treat `offsets[B]` as the logical boundary exactly as an internal consumer would.

Current implementations may temporarily perform broader sanitation while the consumer-responsibility cleanup is staged. That behavior is incidental and is deliberately not part of the public or internal semantic contract. Tests therefore compare logical prefixes and poison inactive capacity rather than asserting tail bytes.

The legacy flattened `map<string, Tensor>` submission surface cannot represent a partition-owning logical ragged boundary and is rejected for stamped networks with such `RaggedNetworkInput`s. Submit a `Batch` containing `RaggedTensor` entries for partition owners. Shared-partition logical inputs may contribute packed `Tensor` values in that same `Batch` because their row partition is supplied by the referenced owner.

## Expression execution

Packed Expression operations currently retain the offsets execution mirror as a structural stage input when they need device-side runtime extent. Active-aware valuewise stages execute only the logical prefix. Bucketed physical operations such as MATMUL (and, after the RMSNorm lifecycle cleanup, bucketed RMSNorm) may choose a larger execution extent; each such consumer owns sanitation of exactly the bucket slack it will physically read. Expression values do not carry a parallel active-row annotation.

## Regression gates

The row-partition runtime tests compile-time check that implementation `Tensor` has no legacy active-row getter, setter, or clearer. End-to-end ragged transformer tests exercise FullyConnected, activations, RMSNorm, DropOut, self-attention, both mixed dense/ragged attention quadrants, residual Add/CustomLayer, segmented reduction, backward training, inference, save/load, and changing packed extents.

Save/load coverage deliberately saves after one host partition and reloads with a different partition, then reuses one placed model across shorter and longer extents. This guards against accidentally serializing runtime host partition state.

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
| Segmented reductions / trailing-dimension ragged Slice | Python `test_segmented_reduction.py` |
| Partition-changing sequence concatenate/slice/gather | `RaggedSequenceConcatenate.*`, `RaggedSequenceSlice.*`, `RaggedGather.*`, Python sequence concatenate/slice/gather tests |
| Architecture/model save-load | `RaggedTensorApi.*` and ragged transformer completeness test |
| TrainingPhase and Python training | ragged transformer TrainingPhase round trip and training integration test |

The final repository-level cutover gate is that the removed values-owned active-row identifier has no contiguous occurrences anywhere in the tree.
