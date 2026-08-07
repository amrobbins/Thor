# Thor einsum permutation/reduction benchmark

`thor_einsum_benchmark` measures the production lowering of:

```text
ijk->ki
```

The benchmark is intentionally narrow. It exists to verify that a zero-copy logical permutation can be reduced through the central `CubReduction` implementation without introducing a global-memory permutation/reduction intermediate.

## Compared paths

1. **`production_cub`**

   ```text
   dense [I,J,K]
       -> zero-copy logical [K,I,J] view
       -> permutation-aware TiledFixedSegment reduction
       -> shared-memory retained-output transpose
       -> dense [K,I]
   ```

   `CubReduction` recognizes the dense physical `[outer=I, reduction=J, inner=K]` source hidden behind the logical view. The tuned tiled reducer traverses that physical layout directly. Finalized retained values are staged through shared memory and written coalesced in the requested dense `[K,I]` order. There is no global permutation/reduction intermediate.

2. **`materializing_reference`**

   ```text
   dense [I,J,K]
       -> reduce J to dense [I,K]
       -> physical transpose/materialize
       -> dense [K,I]
   ```

   This is a historical/reference strategy only. It quantifies the cost of the global-memory intermediate avoided by production; it is not a production candidate.

The benchmark does not expose alternate `CubReduction` stamping APIs or production selection policies. It measures the actual production surface against the materializing reference.

## Cases

The sweep varies reduction extent `J` and physically contiguous retained width `K`:

```text
J = 2, 8, 64, 256, 4096
K = selected widths from 32 through 4096
```

Small `J` makes retained-output traffic important. Large `J` makes source reduction work dominant. The grid includes exact tuned widths and wider component-tiled geometries.

By default each case uses an input of at least `max(512 MiB, 8 x GPU L2)` to reduce accidental L2-resident measurements.

## Build and run

```bash
cmake --build build --target thor_einsum_benchmark -j
./build/thor_einsum_benchmark
```

Useful options:

```bash
./build/thor_einsum_benchmark --target-mib=1024
./build/thor_einsum_benchmark --case=j256
./build/thor_einsum_benchmark --device=1
```

The benchmark currently uses FP32 so the reference intermediate and final output have an unambiguous byte count.

## Output

For each case, the benchmark reports:

- the production and reference CUB reduction paths;
- Expression stage kinds;
- median/best/worst GPU event time;
- minimum logical bytes moved and corresponding GB/s;
- `production_vs_materializing_reference_speedup`, defined as:

```text
materializing_reference_time / production_cub_time
```

Values greater than `1.0` mean production is faster.

`production_cub` minimum traffic is input read + final output write. `materializing_reference` additionally counts the `[I,K]` intermediate write and read. Wall-clock time remains the primary comparison.

Timing samples rotate strategy order to reduce persistent boost/thermal/order bias. Stamping and one-time geometry selection are excluded from timing. CUDA-event synchronization is benchmark-only and does not imply host synchronization in production einsum execution.
