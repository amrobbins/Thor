# Thor fused-kernel census

`thor_fused_kernel_census` is the cache-controlled benchmark used to audit Thor's production fused-expression kernels before changing dispatch or code generation.

## Build and run

Build the Release target:

```bash
cmake --build build-release --target thor_fused_kernel_census
```

Run the full census:

```bash
./build-release/thor_fused_kernel_census | tee fused-kernel-census.csv
```

Useful focused runs:

```bash
./build-release/thor_fused_kernel_census --family=ragged
./build-release/thor_fused_kernel_census --family=ragged_broadcast
./build-release/thor_fused_kernel_census --family=ragged_mixed
./build-release/thor_fused_kernel_census --family=ragged_fp32_compute
./build-release/thor_fused_kernel_census --family=broadcast
./build-release/thor_fused_kernel_census --family=mixed
./build-release/thor_fused_kernel_census --family=indexed
./build-release/thor_fused_kernel_census --case=product_bf16
./build-release/thor_fused_kernel_census --list-cases
```

The default cache-control target is 8x device L2. `--l2-multiple=N`, `--max-rotation-slots=N`, `--warmup-rounds=N`, and `--samples=N` are available for diagnostics, but final comparisons should use the defaults unless there is a specific reason to change them.

## Cache methodology

Each case owns a rotating pool of independently allocated and independently stamped input/output tensors. The pool is sized so that the **reuse distance** before any slot is revisited -- the bytes touched by the other slots in between -- reaches at least 8x L2. The current slot itself is not counted toward that reuse distance.

Each timing event surrounds exactly **one** production fused-kernel launch. The benchmark advances to the next allocation slot between samples, rather than timing a long train of launches and averaging it; this avoids folding host submission gaps into the measured GPU duration.

For small cases, creating enough stamped slots can become unreasonable. The default rotation is capped at 64 slots. If the reuse distance is still smaller than the requested L2 multiple, the benchmark performs a separate device-read pass over an 8x-L2 eviction buffer and synchronizes **before** recording each timing event. The eviction kernel warp-reduces every lane's loaded data into an observable sink so the compiler cannot discard most of the intended reads. Cache eviction is never included in target duration.

This controls accidental **inter-launch** cache residency without defeating legitimate **intra-kernel** cache reuse. In particular, broadcast operands are allowed to remain hot while one fused kernel is executing.

## Byte metrics

The CSV intentionally reports more than one bandwidth metric:

- `effective_bytes` / `effective_gb_s`: one root payload read per output element plus the output write. This preserves the familiar logical/effective interpretation. Broadcast operands are therefore charged once per output element even when one small operand is legitimately reused from cache.
- `compulsory_bytes` / `compulsory_gb_s`: unique target tensor bytes that can be touched once per launch, including the output write and structural metadata that the kernel actually reads. This is the useful lower-bound traffic metric for broadcast cases.
- `model_logical_bytes`: Thor's existing authored logical-work accounting. It is included as a separate model-level diagnostic and should not be interpreted as physical DRAM traffic.

A broadcast case can legitimately report `effective_gb_s` above physical DRAM bandwidth. `compulsory_gb_s` and the cache-control mode make that result interpretable rather than treating the logical number as measured DRAM traffic.

## Dispatch/vectorization metadata

Each row records the actual compiled fused stage:

- `launch_kind`
- `selected_path`
- `explicit_compute_dtype` (`resolved` unless the census deliberately forces a root arithmetic dtype)
- `expected_elements_per_thread` for targeted ordained-path cases
- `elements_per_thread`
- `packet_scalars`
- `input_packet_bytes`
- `output_packet_bytes`
- `max_packet_bytes`
- launch grid/block dimensions
- `registers_per_thread`, `local_bytes_per_thread`, and `static_shared_bytes` queried from the compiled CUDA function
- `device_runtime_extent`
- `runtime_extent_source`
- `pool_touched_bytes` and `reuse_distance_bytes` (plus their L2 ratios), so the cache-control claim is visible in every row

For normal flat kernels, `packet_scalars == elements_per_thread`. For fused tiled transpose, `packet_scalars` reports the tiled-transpose pack width instead.

`input_packet_bytes` is the nominal contiguous value span implied by that packet width for each non-metadata input dtype. It is deliberately described as *nominal*: broadcast operands may load one value and reuse it across a packet, while indexed operands may still issue scalar/indexed loads. `output_packet_bytes` is a direct measure of the output value span owned by one packet/thread.

This means the census should make regressions such as these obvious:

```text
BF16 dense flat:              packet_scalars=8  -> output_packet_bytes=16
FP32 dense flat:              packet_scalars=4  -> output_packet_bytes=16
BF16->FP32 mixed flat:        packet_scalars=4  -> BF16 input span=8, FP32 output=16
```

### Explicit FP32-compute ragged family

`--family=ragged_fp32_compute` is the focused benchmark for the production hole where low-precision storage is promoted to FP32 for arithmetic. Its portable sm89/sm120 packet rule is deliberately storage-based:

```text
BF16/FP16 storage -> BF16/FP16 storage, FP32 compute:
    8 logical elements/thread
    <=16 B per materialized tensor/thread

Any FP32 storage input or output, FP32 compute:
    4 logical elements/thread
    <=16 B per materialized tensor/thread
```

The family includes product-scale unary cases for BF16 and FP16 in both directions across an FP32 storage boundary, product-scale two-input and deeper low-precision-storage cases, medium and launch-limited controls, and partial-tail controls. Targeted rows set `expected_elements_per_thread`; the census aborts before timing if production dispatch does not match that ordained width. `max_packet_bytes` should remain `16` for these ordinary portable cases.

A useful focused run is:

```bash
./build-release/thor_fused_kernel_census --family=ragged_fp32_compute | tee ragged-fp32-compute-census.csv
```
BF16 broadcast:               packet_scalars=2  -> output_packet_bytes=4
BF16 packed ragged aligned:   packet_scalars=8  -> output_packet_bytes=16
BF16 ragged broadcast aligned: packet_scalars=8 -> output_packet_bytes=16
BF16 ragged broadcast awkward: packet_scalars=1 -> output_packet_bytes=2
BF16/FP32 ragged mixed flat:   packet_scalars=4  -> BF16 span=8, FP32 span=16
BF16 storage + FP32 compute:    packet_scalars=8  -> BF16 input/output span=16
BF16 storage -> FP32 storage with FP32 compute: packet_scalars=4 -> BF16 span=8, FP32 span=16
BF16/FP32 ragged broadcast:    packet_scalars=4  -> BF16 span=8, FP32 span=16
Forward `STRIDED_VIEW` is compiled as a runtime storage alias, not retained as an indexed fused node.
Its reported packet width therefore reflects the ordinary flat/specialized-broadcast kernel selected for the alias layout.
Genuinely index-aware controls such as `TAKE_ALONG_AXIS` remain labeled as indexed gathers.
Indexed gather/awkward view:    packet_scalars=1  -> scalar fallback
```

Those are observations to verify from the current compiler, not benchmark hard-coded expectations. Future optimizations should change the emitted metadata naturally.

## Census coverage

The default suite covers:

- homogeneous BF16, FP16, and FP32 flat unary, binary, and deeper fused trees;
- aligned and awkward-tail dense shapes;
- BF16/FP32 mixed storage/output combinations;
- row, column, outer, and scalar-tensor broadcasts;
- mixed-dtype broadcast;
- packed device-runtime-extent ragged valuewise kernels, including homogeneous and mixed dtype;
- mixed ragged packetization controls at small, medium, awkward, and product-transformer scales;
- explicit FP32-compute ragged controls proving that BF16/FP16 storage still owns vector8 packets when both sides remain low precision, while any FP32 storage boundary uses vector4; the product-scale matrix covers BF16/FP16 -> same dtype, BF16/FP16 -> FP32, and FP32 -> BF16/FP16;
- exact product-transformer scale `T = 128 * 819 = 104832` at widths 80, 128, 256, 384, and 512;
- ragged broadcast at those product-transformer widths, including explicit BF16/FP16-storage + FP32-compute aligned cases and BF16/FP32 mixed row/scalar/output-dtype variants;
- non-transformer ragged-broadcast sweeps at `T = 512`, `8192`, and `16385`, including packet-aligned widths 8, 24, 64, 80, 96, 128, and 192;
- explicit row-boundary rejection cases at widths 7, 15, 79, 81, 127, and 129 where `T = 8192` makes total numel divisible by 8 even though one ragged value is not packet aligned;
- scalar-broadcast controls at small, medium, and odd-`T` ragged extents, plus medium-`T` per-row scalar and two-broadcast-operand patterns;
- representative non-transformer FP16 aligned, scalar-broadcast, and awkward-width cases alongside the broader BF16 sweep;
- small, medium, awkward, and product-scale packed-ragged shapes;
- index-aware forward strided views whose innermost physical span is contiguous, including BF16/FP16/FP32, first-half and second-half views, product-scale and medium-scale shapes;
- exact ragged-transformer SwiGLU-style sibling views over `[T,2H]` storage (`[T,H]` value and gate halves);
- strided-alias controls for aligned and misaligned view offsets, awkward row widths, non-unit inner stride, and unusual outer row stride;
- a genuine `TAKE_ALONG_AXIS` gather control that remains index-aware;
- a `TAKE_ALONG_AXIS` runtime gather control that must remain on the scalar/index-mapped path;
- representative fused tiled-transpose cases as a regression control for the previous performance suite.

The timed plans are required to contain exactly one production `FusedKernel` stage. If a case starts lowering through another operation family or multiple stages, the census fails rather than silently timing a different workload.

## FUSED-GATE regression invariant

The benchmark-backed fused fast paths are enforced during equation compilation, not only by this census.
`CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread()` and
`fusedGateRequiredSpecializedBroadcastElementsPerThread()` independently classify the ordinary layouts that
have an ordained packet width. `EquationCompiler` compares that requirement with the dispatch-selected width
before NVRTC compilation and throws a `FUSED-GATE` error if dispatch has silently narrowed the stage.

The current gate covers:

- dense homogeneous BF16/FP16 flat: at least 8 scalars/thread;
- dense FP32 and benchmarked BF16/FP16↔FP32 mixed flat: at least 4 scalars/thread;
- packed-ragged homogeneous BF16/FP16 storage flat: at least 8 scalars/thread, including explicit FP32 compute;
- packed-ragged FP32-storage and benchmarked BF16/FP16↔FP32 storage-boundary flat: at least 4 scalars/thread;
- dense homogeneous BF16/FP16 specialized broadcast: at least 2 scalars/thread;
- packet-aligned ragged homogeneous BF16/FP16 storage broadcast: at least 8 scalars/thread, including explicit FP32 compute;
- packet-aligned ragged FP32-storage and benchmarked FP32 mixed broadcast: at least 4 scalars/thread.

The gate intentionally does not impose a wider path on genuine gathers/scatters, transpose/index-aware
operations, awkward ragged broadcast row cycles, runtime-scalar/unbenchmarked layouts, or BF16↔FP16-only mixed
traffic. Those remain explicit exceptions rather than silently weakening the ordinary fast-path invariant.
