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
- `elements_per_thread`
- `packet_scalars`
- `input_packet_bytes`
- `output_packet_bytes`
- launch grid/block dimensions
- `registers_per_thread`, `local_bytes_per_thread`, and `static_shared_bytes` queried from the compiled CUDA function
- `device_runtime_extent`
- `runtime_extent_source`
- `pool_touched_bytes` and `reuse_distance_bytes` (plus their L2 ratios), so the cache-control claim is visible in every row

For normal flat kernels, `packet_scalars == elements_per_thread`. For fused tiled transpose, `packet_scalars` reports the tiled-transpose pack width instead.

`input_packet_bytes` is the nominal contiguous value span implied by that packet width for each non-metadata input dtype. It is deliberately described as *nominal*: broadcast operands may load one value and reuse it across a packet, while indexed operands may still issue scalar/indexed loads. `output_packet_bytes` is a direct measure of the output value span owned by one packet/thread.

This means the census should make regressions such as these obvious:

```text
BF16 dense flat:       packet_scalars=8  -> output_packet_bytes=16
FP32 dense flat:       packet_scalars=4  -> output_packet_bytes=16
BF16->FP32 mixed flat: packet_scalars=4  -> BF16 input span=8, FP32 output=16
BF16 broadcast:        packet_scalars=2  -> output_packet_bytes=4
BF16 packed ragged:    packet_scalars=1  -> output_packet_bytes=2
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
- exact product-transformer scale `T = 128 * 819 = 104832` at widths 80, 128, 256, 384, and 512;
- ragged broadcast at those product-transformer widths;
- small, medium, awkward, and product-scale packed-ragged shapes;
- index-aware strided views whose innermost physical span is contiguous, including a deliberately misaligned BF16 view;
- representative fused tiled-transpose cases as a regression control for the previous performance suite.

The timed plans are required to contain exactly one production `FusedKernel` stage. If a case starts lowering through another operation family or multiple stages, the census fails rather than silently timing a different workload.
