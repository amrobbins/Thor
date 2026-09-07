# Ragged row-partition runtime model

The repository-level public support matrix and qualification checklist live in
[`ragged_support_contract.md`](ragged_support_contract.md). This document describes
the final RP1-RP7 physical ownership model for rank-1 ragged row partitions.

## Logical representation

Thor separates ragged values, partition identity, graph topology, and runtime
partition bytes:

```text
RaggedTensor
  values: Tensor
  rowPartitionId: RowPartitionId
  rowPartitionToken: Tensor        # graph-local topology only
  descriptor: RowPartitionDescriptor
```

The `rowPartitionToken` exists because Thor's API graph is tensor-edge based. It
lets a layer declare that it consumes partition `P`, but it is **not** a promise
that a device `[B+1]` offsets tensor exists. `RaggedTensor::getOffsets()` remains
a compatibility alias for this graph-local token; semantic partition equality is
`sharesPartitionWith()` / `RowPartitionId`, not Tensor equality.

A partition-preserving operation carries the exact same `RowPartitionId` and
partition token to its output. A partition-changing operation creates or adopts a
new `RowPartitionId` whose authoritative host partition is established before its
values execute.

## Runtime source of truth

For batch size `B`, the complete host vector

```text
hostOffsets[0 : B + 1]
```

is the only semantic source of truth for a live partition. It must satisfy the
canonical monotonic/capacity contract. Thor derives:

```text
activeValueCount   = hostOffsets[B]
maxActiveRowLength = max(hostOffsets[i + 1] - hostOffsets[i])
```

from that publication. Those quantities cannot be published independently.
There are no generation/revision counters and generic writes to a device tensor
do not redefine the logical partition.

A supported executing partition must have authoritative host offsets. CPU offsets
supplied at an external logical boundary can establish them immediately because
the bytes are already host-resident. A GPU-only offsets payload without matching
host partition state is rejected; Thor does not perform a D2H synchronization to
make device bytes authoritative. Once structural metadata has an authoritative
producer, physical consumers must reuse that state rather than independently
reconstructing the same partition from payload or source metadata.

## Requirement-driven physical representations

A logical partition does not inherently own a device offsets tensor. During
placement, each physical consumer declares exactly which information it needs via
`RaggedPartitionRequirement`:

```text
NONE
HOST_EXTENT
DEVICE_ACTIVE_COUNT   # managed [1]
DEVICE_OFFSETS        # managed [B+1]
```

Requirements are aggregated per `RowPartitionId`, but each consumer is wired to
its own requested representation. For example, a valuewise loss may receive the
managed `[1]` active-count carrier while a segmented reduction over the same
partition independently receives `[B+1]` offsets.

`HOST_EXTENT` is metadata carried with a values path for host-directed physical
shape selection; it does not make the carrier payload an offsets tensor.
`DEVICE_ACTIVE_COUNT` is a true scalar device input. `DEVICE_OFFSETS` is the full
row-boundary representation and exists only when some physical consumer genuinely
needs row starts/ends.

The default requirement is `NONE`. A new physical ragged consumer therefore must
explicitly declare the partition bytes it consumes rather than inheriting a
conservative full-offset allocation.

## Network boundaries

A fresh public `RaggedNetworkInput` exposes one logical input name whose payload is
a `PhysicalRaggedTensor`/logical ragged batch value. Its packed values are the only
public graph input tensor. The partition token is represented internally by a
non-external graph-local input used for topology and is never stamped or dataset-
bound.

At batch submission, the partition-owning input publishes complete authoritative
host offsets. Placement-created managed `[1]` and `[B+1]` inputs are populated from
that host state only if aggregate requirements requested them. A shared-partition
`RaggedNetworkInput(..., partition=source)` contributes only another values stream
and reuses the owner's `RowPartitionId`.

A fresh `RaggedNetworkOutput` likewise has only a physical values output. Merely
exposing a ragged result does not promote its partition to `HOST_EXTENT` or force a
device offsets allocation. `inferLogical()` reconstructs the returned offsets view
from the authoritative host partition for that output's `RowPartitionId`; no D2H
read of a device offsets tensor is required.

The legacy flattened `map<string, Tensor>` submission surface cannot represent a
partition-owning ragged logical boundary and is rejected for networks that contain
one. Submit a logical `Batch` with ragged values instead.

## Partition-preserving operations

A partition-preserving operation changes packed values while retaining partition
`P`:

```text
(values A, P) -> layer -> (values B, P)
```

Examples include ordinary activations, FullyConnected, normalization, DropOut,
TypeConverter, gradient-control layers, trailing shape operations, trailing-axis
Concatenate/Slice, and ragged CustomLayer operations.

These APIs may still call `getOffsets()` as a graph-topology compatibility handle,
but semantic compatibility uses `RowPartitionId`. Placement resolves that token to
whatever physical representation the consumer declared.

## Internally-created partitions

Partition-changing operations establish their output partition on the host before
GPU values execute. The stamped network records how each internal `RowPartitionId`
is derived so host state can be resolved recursively for every batch.

### `RaggedSequenceConcatenate`

For source partitions `P_i`, the CPU derives the output host partition `Q` by
summing corresponding row lengths. The physical concatenate implementation outputs
values only and consumes the source partitions needed to locate input rows. It does
not force any physical representation of `Q`; downstream requirements decide whether
`Q` gets `HOST_EXTENT`, `[1]`, `[B+1]`, or nothing.

### `RaggedSequenceSlice`

The CPU derives clipped output host offsets `Q` from the authoritative source
partition. Slice compaction itself genuinely needs the new row starts, so its own
placement requirement contributes `DEVICE_OFFSETS` for `Q`. The managed `[B+1]`
representation is therefore an **input** to the values-only Slice implementation,
not a GPU-produced semantic partition output.

### `RaggedGather`

Gather does not derive a new partition from source data. Its output adopts the
already-authoritative partition of the ragged indices input.

Thor deliberately does not support a partition-producing operation whose row
boundaries depend on runtime device values. Such an operation would require a D2H
synchronization to establish authoritative host offsets. The former `RaggedFilter`
surface was removed for this reason.

## Inactive packed capacity

Storage in

```text
[activeValueCount, maxTotalValues)
```

is undefined. It is not semantic zero-padding and may contain NaN, Inf, stale bytes,
or arbitrary data.

Active-aware kernels execute only the logical prefix. A physical consumer that
intentionally reads a larger bucket (for example a selected packed GEMM shape) owns
sanitation of exactly the extra region it reads immediately before the read. That
sanitized slack is not persistent partition state and becomes undefined again after
the consumer.

The same rule applies at network boundaries: callers must use the returned logical
partition to determine the valid prefix and must not assign meaning to capacity tail
bytes.

## Expression execution and backward

Ragged expression stages carry an explicit runtime-extent source. Valuewise stages
normally consume `DEVICE_ACTIVE_COUNT`; segmented/row-indexed stages consume
`DEVICE_OFFSETS`; packed shape-selection paths may additionally require
`HOST_EXTENT`. The compiler and API placement requirement must agree on that source.

Backward stages follow the same rule independently of forward. A forward operation
that only needs `[1]` can strengthen to `[B+1]` during training when a genuine
backward operation is segmented—for example a row-aware parameter reduction. This is
an explicit backward requirement, not compatibility fallback from full offsets to a
scalar carrier.

## Lifetime, save/load, and composition

Authoritative host offsets are per-batch runtime state and are not serialized.
Model/architecture files persist structural descriptor/topology information only.
After load, a newly placed network receives authoritative host partitions from new
batch submissions and must accept a different valid partition from the one used
before save.

`RowPartitionId` is graph-local. Save/load and subgraph/phase cloning reconstruct or
remap partition identity together with values topology. The hidden partition token is
also remapped so structural consumers in a cloned graph reference the destination
partition rather than the source graph's token.

Composed networks splice logical output/input boundaries before physical stamping;
partition identity is preserved/remapped as part of that graph composition rather
than by chaining a physical offsets output between networks.

## Regression gates

The repository-level closure gate is:

```bash
cmake --build <build-dir> --target check-ragged-support-contract
```

It gathers the public boundary, structural, learning, loss, metric, Softmax,
attention, CTC, adapter, and partition-changing regression suites. In particular,
it exercises both offset widths, poisoned inactive capacity, empty/all-empty rows,
short-long-short reuse, save/load with changed partitions, clone remapping, forward
and backward, partial batches, weighted/ratio metric aggregation, no-contribution
extrema, ordinary-vs-segmented Softmax semantics, and intentional unsupported dtype
rejection.

The heavier retained ragged Conv1D performance/timing gate remains separate.
