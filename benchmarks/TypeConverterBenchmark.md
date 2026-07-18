# TypeConverter GPU benchmark

`thor_type_converter_benchmark` measures Thor's real `TypeConverter::convertType` launch path over a logarithmic tensor-size sweep. It is an opt-in target so normal Thor builds and test runs do not execute performance measurements.

## Build

Use a Release build for meaningful results:

```bash
cmake --build build --target thor_type_converter_benchmark -j
```

## Run

The default logarithmic sweep is 2K, 8K, 32K, 128K, 512K, 2M, 8M, 32M, 128M, and 256M elements. The K/M suffixes are binary, so the upper endpoint is 268,435,456 elements.

The default `common` suite covers the conversions most relevant to Thor training plus representative integer and boolean conversions:

```bash
./build/thor_type_converter_benchmark --csv=type_converter_common.csv
```

All ordered floating-point storage-type pairs:

```bash
./build/thor_type_converter_benchmark \
  --suite=ml \
  --csv=type_converter_ml.csv
```

Every supported ordered storage-type pair:

```bash
./build/thor_type_converter_benchmark \
  --suite=all \
  --csv=type_converter_all.csv
```

A focused tuning run can select one or more pairs and a custom size sweep:

```bash
./build/thor_type_converter_benchmark \
  --pair=fp32:bf16 \
  --pair=bf16:fp32 \
  --sizes=2K,8K,32K,128K,512K,2M,8M,32M,128M,256M \
  --samples=11 \
  --csv=fp32_bf16.csv
```

In-place conversions are available separately because they use a different multi-launch implementation and are not the normal TypeConverter layer path:

```bash
./build/thor_type_converter_benchmark --suite=ml --mode=both --csv=type_converter_both.csv
```

Run `--help` for all controls.

## Avoiding L2-resident measurements

The benchmark is deliberately not a loop over one input tensor.

For each case it:

1. Queries `cudaDevAttrL2CacheSize`.
2. Allocates a separate eviction buffer sized to at least 4x reported L2 capacity, with a 64 MiB minimum.
3. Initializes source storage with deterministic, hashed, exactly representable values rather than a constant or short repeating byte pattern.
4. Runs a GPU read-modify-write sweep over the entire eviction buffer before every timed sample.
5. Rotates the timed launches through disjoint source and destination slots, aligned and padded to 256 bytes, so a launch never reuses a cache line touched by an earlier launch in the same sample.
6. Places CUDA timing events after the eviction sweep and around only the TypeConverter launches.

Small conversions therefore retain launch-latency sensitivity without repeatedly loading the same L2-resident tensor. Large conversions naturally exceed L2 and stream through GPU main memory; the same eviction step also prevents their first portion from beginning warm on repeated samples.

The conversion working set is capped independently of the eviction allocation. The default cap is 4 GiB so every heterogeneous storage-type pair can fit one 256M-element out-of-place slot when that much GPU memory is available. If a case still cannot fit while preserving the configured GPU-memory reserve, it is reported as skipped rather than shrinking the tensor.

## Output

CSV rows include:

- median, p10, and p90 microseconds per TypeConverter call;
- median nanoseconds per element;
- effective read-plus-write GB/s;
- input-only and output-only GB/s;
- launches per timed sample;
- logical tensor bytes, physical rotating working-set bytes, reported L2 bytes, and eviction-buffer bytes.

Progress and GPU metadata are written to stderr so stdout remains valid CSV when `--csv` is omitted.
