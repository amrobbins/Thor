#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr uint64_t MIN_INPUT_BYTES = 512ULL * MIB;
constexpr uint64_t L2_WORKING_SET_MULTIPLE = 8;
constexpr int WARMUP_ITERATIONS = 3;
constexpr int TIMING_SAMPLES = 5;
constexpr int TIMED_ITERATIONS_PER_SAMPLE = 4;

struct ReductionShape {
    const char* name;
    std::vector<uint64_t> reduction_dimensions;
    uint64_t inner_size;
};

bool isFocusedArgX4Shape(std::string_view name) {
    constexpr std::string_view prefixes[] = {"r64_d", "r256_d", "r1024_d"};
    constexpr std::string_view widths[] = {"128", "256", "512", "1024", "2048", "4096", "65536"};
    for (std::string_view prefix : prefixes) {
        for (std::string_view width : widths) {
            if (name == std::string(prefix) + std::string(width)) {
                return true;
            }
        }
    }
    return false;
}

bool isFocusedArgX4AwkwardShape(std::string_view name) {
    constexpr std::string_view prefixes[] = {"r64_d", "r256_d", "r1024_d"};
    constexpr std::string_view widths[] = {"4096", "4097", "8192", "8193", "65536", "65537"};
    for (std::string_view prefix : prefixes) {
        for (std::string_view width : widths) {
            if (name == std::string(prefix) + std::string(width)) {
                return true;
            }
        }
    }
    return false;
}

bool isFocusedArgX4DType(DataType dtype) {
    return dtype == DataType::FP8_E4M3 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* quantity) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string(quantity) + " overflowed uint64_t.");
    }
    return lhs * rhs;
}

uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

const char* dataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP32:
            return "fp32";
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return "fp64";
#endif
        default:
            return "unsupported";
    }
}

const char* operationName(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
            return "sum";
        case CubReductionOp::Mean:
            return "mean";
        case CubReductionOp::Min:
            return "min";
        case CubReductionOp::Max:
            return "max";
        case CubReductionOp::Product:
            return "product";
        case CubReductionOp::L1Norm:
            return "l1";
        case CubReductionOp::L2Norm:
            return "l2";
    }
    return "unknown";
}

const char* operationName(CubArgReductionOp op) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return "argmin_index";
        case CubArgReductionOp::ArgMax:
            return "argmax_index";
    }
    return "unknown_arg";
}

const char* pathName(CubReductionPath path) {
    switch (path) {
        case CubReductionPath::DeviceTransformReduce:
            return "device_transform";
        case CubReductionPath::ContiguousFixedSegment:
            return "contiguous_segment";
        case CubReductionPath::TiledFixedSegment:
            return "tiled_segment";
        case CubReductionPath::StridedFixedSegment:
            return "strided_segment";
        case CubReductionPath::OffsetSegmented:
            return "offset_segmented";
    }
    return "unknown";
}

int nextPowerOfTwoWarps(uint64_t required) {
    int warps = 1;
    while (warps < 8 && static_cast<uint64_t>(warps) < required) {
        warps *= 2;
    }
    return static_cast<uint64_t>(warps) >= required ? warps : 0;
}

int groupedFullRowWarps(uint64_t inner_size, DataType input_dtype) {
    constexpr uint64_t components_per_warp = 512;
    constexpr uint64_t stage_bytes_per_warp = 2048;
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));
    const uint64_t component_warps = ceilDiv(inner_size, components_per_warp);
    const uint64_t stage_warps = ceilDiv(checkedMultiply(inner_size, element_bytes, "benchmark row bytes"),
                                         stage_bytes_per_warp);
    return nextPowerOfTwoWarps(std::max(component_warps, stage_warps));
}

const char* tiledStrategyName(uint64_t inner_size, DataType input_dtype) {
    if (inner_size <= 32) {
        return "async_full_row_narrow";
    }
    if (inner_size < 64) {
        return "async_full_row_x2";
    }
    if (inner_size == 64) {
        return "vector_direct_x2";
    }
    if (inner_size < 128) {
        return "async_full_row_x4";
    }
    if (inner_size == 128) {
        return "vector_direct_x4";
    }
    if (inner_size < 256) {
        return "async_full_row_x8";
    }
    if (inner_size == 256) {
        return "vector_direct_x8";
    }
    if (inner_size < 512) {
        if (input_dtype == DataType::FP64 && inner_size > 256) {
            return "direct_component_tiled";
        }
        return "async_full_row_x16";
    }
    if (inner_size == 512) {
        return "vector_direct_x16";
    }
    if (inner_size == 1024) {
        return "vector_direct_group2_x16";
    }
    if (inner_size == 2048) {
        return "vector_direct_group4_x16";
    }
    if (inner_size == 4096) {
        return "vector_direct_group8_x16";
    }
    if (inner_size < 4096) {
        switch (groupedFullRowWarps(inner_size, input_dtype)) {
            case 2:
                return "async_full_row_group2_x16";
            case 4:
                return "async_full_row_group4_x16";
            case 8:
                return "async_full_row_group8_x16";
            default:
                break;
        }
    }
    if (inner_size > 4096) {
        return inner_size % 4096 == 0 ? "vector_direct_block_shards_x16"
                                      : "alignment_safe_vectorized_shaped_block_shards_x16";
    }
    return "direct_component_tiled";
}

const char* strategyName(const StampedCubReduction& reduction) {
    if (reduction.getPath() == CubReductionPath::TiledFixedSegment) {
        return tiledStrategyName(reduction.getGeometry().inner_size, reduction.getInputDataType());
    }
    return pathName(reduction.getPath());
}

const char* tiledArgStrategyName(uint64_t inner_size) {
    if (inner_size == 64) {
        return "vector_direct_x2";
    }
    if (inner_size == 128) {
        return "vector_direct_x4";
    }
    if (inner_size == 256) {
        return "vector_direct_group2_x4";
    }
    if (inner_size == 512) {
        return "vector_direct_group4_x4";
    }
    if (inner_size == 1024) {
        return "vector_direct_group8_x4";
    }
    if (inner_size > 1024 && inner_size % 1024 == 0) {
        return "vector_direct_block_shards_x4";
    }
    if (inner_size > 4096) {
        return "alignment_safe_vectorized_block_shards_x4";
    }
    if (inner_size <= 2) {
        return "direct_tiled_row_lanes16";
    }
    if (inner_size <= 4) {
        return "direct_tiled_row_lanes8";
    }
    if (inner_size <= 8) {
        return "direct_tiled_row_lanes4";
    }
    if (inner_size <= 16) {
        return "direct_tiled_row_lanes2";
    }
    return "direct_tiled_row_lanes1";
}

const char* strategyName(const StampedCubArgReduction& reduction) {
    if (reduction.getPath() == CubReductionPath::TiledFixedSegment) {
        return tiledArgStrategyName(reduction.getGeometry().inner_size);
    }
    return pathName(reduction.getPath());
}

std::vector<uint64_t> makeInputDimensions(uint64_t outer_size, const ReductionShape& shape) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(shape.reduction_dimensions.size() + 2);
    dimensions.push_back(outer_size);
    dimensions.insert(dimensions.end(), shape.reduction_dimensions.begin(), shape.reduction_dimensions.end());
    if (shape.inner_size > 1) {
        dimensions.push_back(shape.inner_size);
    }
    return dimensions;
}

std::vector<uint32_t> makeReductionAxes(const ReductionShape& shape) {
    std::vector<uint32_t> axes;
    axes.reserve(shape.reduction_dimensions.size());
    for (uint32_t i = 0; i < shape.reduction_dimensions.size(); ++i) {
        axes.push_back(i + 1);
    }
    return axes;
}

uint64_t reductionElementsPerOuter(const ReductionShape& shape) {
    uint64_t elements = shape.inner_size;
    for (uint64_t dimension : shape.reduction_dimensions) {
        elements = checkedMultiply(elements, dimension, "benchmark elements per outer slice");
    }
    return elements;
}

void runCase(const ReductionShape& shape,
             DataType dtype,
             CubReductionOp op,
             uint64_t target_input_bytes,
             Stream& stream,
             const TensorPlacement& gpu_placement) {
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
    const uint64_t elements_per_outer = reductionElementsPerOuter(shape);
    const uint64_t target_elements = ceilDiv(target_input_bytes, element_bytes);
    const uint64_t outer_size = std::max<uint64_t>(1, ceilDiv(target_elements, elements_per_outer));

    const std::vector<uint64_t> dimensions = makeInputDimensions(outer_size, shape);
    const std::vector<uint32_t> axes = makeReductionAxes(shape);
    Tensor input(gpu_placement, TensorDescriptor(dtype, dimensions));

    // Initialization is outside the timed interval. The benchmark deliberately makes each input much larger than L2,
    // so a repeated reduction cannot become an L2-resident benchmark after warm-up.
    checkCuda(cudaMemsetAsync(input.getMemPtr<void>(), 0, input.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(input)");
    stream.synchronize();

    std::shared_ptr<StampedCubReduction> stamped = CubReduction(op, axes).stamp(input, stream);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        stamped->runOn(stream);
    }
    stream.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::vector<double> sample_ms;
    sample_ms.reserve(TIMING_SAMPLES);
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(start)");
        for (int i = 0; i < TIMED_ITERATIONS_PER_SAMPLE; ++i) {
            stamped->runOn(stream);
        }
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        sample_ms.push_back(static_cast<double>(elapsed_ms) / TIMED_ITERATIONS_PER_SAMPLE);
    }
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::sort(sample_ms.begin(), sample_ms.end());
    const double best_ms = sample_ms.front();
    const double median_ms = sample_ms[sample_ms.size() / 2];
    const double worst_ms = sample_ms.back();
    const Tensor output = stamped->getOutputTensor();
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output.getArraySizeInBytes();
    const double logical_gb_per_second = static_cast<double>(logical_bytes) / (median_ms * 1.0e6);

    std::cout << shape.name << ',' << dataTypeName(dtype) << ',' << operationName(op) << ','
              << pathName(stamped->getPath()) << ',' << strategyName(*stamped) << ',' << outer_size << ','
              << stamped->getGeometry().reduction_size << ',' << stamped->getGeometry().inner_size << ','
              << input.getArraySizeInBytes() << ',' << output.getArraySizeInBytes() << ',' << std::fixed
              << std::setprecision(4) << median_ms << ',' << best_ms << ',' << worst_ms << ',' << std::setprecision(2)
              << logical_gb_per_second << '\n';
}

void runArgCase(const ReductionShape& shape,
                DataType dtype,
                CubArgReductionOp op,
                uint64_t target_input_bytes,
                Stream& stream,
                const TensorPlacement& gpu_placement) {
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
    const uint64_t elements_per_outer = reductionElementsPerOuter(shape);
    const uint64_t target_elements = ceilDiv(target_input_bytes, element_bytes);
    const uint64_t outer_size = std::max<uint64_t>(1, ceilDiv(target_elements, elements_per_outer));

    const std::vector<uint64_t> dimensions = makeInputDimensions(outer_size, shape);
    const std::vector<uint32_t> axes = makeReductionAxes(shape);
    Tensor input(gpu_placement, TensorDescriptor(dtype, dimensions));

    // All-zero input makes every ARG comparison take the tie-breaking path. Use finite randomized values instead so
    // the timed kernel measures the normal min/max candidate path. Initialization remains entirely outside timing.
    input.fillRandom(-100.0, 100.0, stream);
    stream.synchronize();

    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = DataType::UINT32;
    std::shared_ptr<StampedCubArgReduction> stamped = CubArgReduction(op, axes, outputs).stamp(input, stream);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        stamped->runOn(stream);
    }
    stream.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::vector<double> sample_ms;
    sample_ms.reserve(TIMING_SAMPLES);
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(start)");
        for (int i = 0; i < TIMED_ITERATIONS_PER_SAMPLE; ++i) {
            stamped->runOn(stream);
        }
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        sample_ms.push_back(static_cast<double>(elapsed_ms) / TIMED_ITERATIONS_PER_SAMPLE);
    }
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::sort(sample_ms.begin(), sample_ms.end());
    const double best_ms = sample_ms.front();
    const double median_ms = sample_ms[sample_ms.size() / 2];
    const double worst_ms = sample_ms.back();
    const uint64_t value_output_bytes = stamped->getValueOutputTensor().has_value()
                                            ? stamped->getValueOutputTensor()->getArraySizeInBytes()
                                            : 0;
    const uint64_t index_output_bytes = stamped->getIndexOutputTensor().has_value()
                                            ? stamped->getIndexOutputTensor()->getArraySizeInBytes()
                                            : 0;
    const uint64_t output_bytes = value_output_bytes + index_output_bytes;
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output_bytes;
    const double logical_gb_per_second = static_cast<double>(logical_bytes) / (median_ms * 1.0e6);

    std::cout << shape.name << ',' << dataTypeName(dtype) << ',' << operationName(op) << ','
              << pathName(stamped->getPath()) << ',' << strategyName(*stamped) << ',' << outer_size << ','
              << stamped->getGeometry().reduction_size << ',' << stamped->getGeometry().inner_size << ','
              << input.getArraySizeInBytes() << ',' << output_bytes << ',' << std::fixed << std::setprecision(4)
              << median_ms << ',' << best_ms << ',' << worst_ms << ',' << std::setprecision(2)
              << logical_gb_per_second << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    bool focused_arg_x4 = false;
    bool focused_arg_x4_awkward = false;
    if (argc == 2 && std::string_view(argv[1]) == "--arg-x4-focused") {
        focused_arg_x4 = true;
    } else if (argc == 2 && std::string_view(argv[1]) == "--arg-x4-awkward-focused") {
        focused_arg_x4_awkward = true;
    } else if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        std::cout << "Usage: " << argv[0] << " [--arg-x4-focused|--arg-x4-awkward-focused]\n"
                  << "  --arg-x4-focused          Run ARGMIN only for FP8 E4M3/FP16/FP32, R=64/256/1024, and "
                     "D=128/256/512/1024/2048/4096/65536.\n"
                  << "  --arg-x4-awkward-focused  Run ARGMIN exact/awkward A/B pairs for FP8 E4M3/FP16/FP32, "
                     "R=64/256/1024, and D=4096/4097/8192/8193/65536/65537.\n";
        return EXIT_SUCCESS;
    } else if (argc != 1) {
        std::cerr << "Unknown benchmark arguments. Use --help for supported options.\n";
        return EXIT_FAILURE;
    }

    try {
        int device_count = 0;
        checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "CubReductionBenchmark requires a CUDA GPU.\n";
            return EXIT_FAILURE;
        }

        constexpr int device = 0;
        ScopedGpu scoped_gpu(device);

        int l2_cache_bytes = 0;
        checkCuda(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, device),
                  "cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)");
        if (l2_cache_bytes <= 0) {
            throw std::runtime_error(
                "CUDA reported a non-positive L2 cache size; refusing to run a cache-sensitive benchmark.");
        }

        const uint64_t target_input_bytes =
            std::max<uint64_t>(MIN_INPUT_BYTES, checkedMultiply(static_cast<uint64_t>(l2_cache_bytes),
                                                                 L2_WORKING_SET_MULTIPLE,
                                                                 "L2-sized benchmark working set"));

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        if (target_input_bytes > free_bytes / 3) {
            throw std::runtime_error(
                "Insufficient free GPU memory for the cache-cold reduction benchmark. The benchmark intentionally "
                "requires an input working set at least 8x L2 and will not shrink into an L2-resident measurement.");
        }

        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

        std::cout << "# device=" << properties.name << " l2_bytes=" << l2_cache_bytes
                  << " target_input_bytes=" << target_input_bytes << " free_bytes=" << free_bytes
                  << " total_bytes=" << total_bytes << " target_over_l2=" << std::fixed << std::setprecision(2)
                  << static_cast<double>(target_input_bytes) / l2_cache_bytes << '\n';
        std::cout << "# Each timed reduction reads an input >= target_input_bytes. Because the input is >= 8x L2, "
                     "successive iterations cannot benchmark an L2-resident working set.\n";
        std::cout << "# timing_samples=" << TIMING_SAMPLES
                  << " timed_iterations_per_sample=" << TIMED_ITERATIONS_PER_SAMPLE
                  << " reported_time=median\n";
        std::cout << "# argmin_index/argmax_index benchmark the production index-only UINT32 path using randomized "
                     "finite input initialized outside the timed interval.\n";
        if (focused_arg_x4) {
            std::cout << "# mode=arg_x4_focused operations=argmin_index "
                         "dtypes=fp8_e4m3|fp16|fp32 reductions=64|256|1024 "
                         "inners=128|256|512|1024|2048|4096|65536\n";
        } else if (focused_arg_x4_awkward) {
            std::cout << "# mode=arg_x4_awkward_focused operations=argmin_index "
                         "dtypes=fp8_e4m3|fp16|fp32 reductions=64|256|1024 "
                         "inners=4096|4097|8192|8193|65536|65537\n";
        }
        std::cout << "shape,dtype,operation,path,strategy,outer,reduction,inner,input_bytes,output_bytes,median_ms,"
                     "best_ms,worst_ms,logical_GBps\n";

        // Primary width sweep: exercise every ownership/vectorization boundary, including one value on each side.
        // The large-width section probes the attention-reducer-style geometry transition from several outputs per block
        // to 2-warp, 4-warp, and finally one-CTA-per-output full-row groups, plus the first fallback width above 4096.
        const std::vector<ReductionShape> shapes = {
            {"r256_d1", {256}, 1},
            {"r256_d2", {256}, 2},
            {"r256_d3", {256}, 3},
            {"r256_d4", {256}, 4},
            {"r256_d7", {256}, 7},
            {"r256_d8", {256}, 8},
            {"r256_d9", {256}, 9},
            {"r256_d15", {256}, 15},
            {"r256_d16", {256}, 16},
            {"r256_d17", {256}, 17},
            {"r256_d31", {256}, 31},
            {"r256_d32", {256}, 32},
            {"r256_d33", {256}, 33},
            {"r256_d63", {256}, 63},
            {"r256_d64", {256}, 64},
            {"r256_d65", {256}, 65},
            {"r256_d127", {256}, 127},
            {"r256_d128", {256}, 128},
            {"r256_d129", {256}, 129},
            {"r256_d255", {256}, 255},
            {"r256_d256", {256}, 256},
            {"r256_d257", {256}, 257},
            {"r256_d511", {256}, 511},
            {"r256_d512", {256}, 512},
            {"r256_d513", {256}, 513},
            {"r256_d768", {256}, 768},
            {"r256_d1023", {256}, 1023},
            {"r256_d1024", {256}, 1024},
            {"r256_d1025", {256}, 1025},
            {"r256_d1536", {256}, 1536},
            {"r256_d2047", {256}, 2047},
            {"r256_d2048", {256}, 2048},
            {"r256_d2049", {256}, 2049},
            {"r256_d3072", {256}, 3072},
            {"r256_d4095", {256}, 4095},
            {"r256_d4096", {256}, 4096},
            {"r256_d4097", {256}, 4097},
            {"r256_d6144", {256}, 6144},
            {"r256_d8191", {256}, 8191},
            {"r256_d8192", {256}, 8192},
            {"r256_d8193", {256}, 8193},
            {"r256_d16384", {256}, 16384},
            {"r256_d16385", {256}, 16385},
            {"r256_d65536", {256}, 65536},
            {"r256_d65537", {256}, 65537},
            {"r256_d262144", {256}, 262144},
            {"r256_d262145", {256}, 262145},

            // Reduction-length sensitivity for representative async, vector-direct, and fallback widths.
            {"r64_d33", {64}, 33},
            {"r1024_d33", {1024}, 33},
            {"r64_d128", {64}, 128},
            {"r1024_d128", {1024}, 128},
            {"r64_d255", {64}, 255},
            {"r1024_d255", {1024}, 255},
            {"r64_d256", {64}, 256},
            {"r1024_d256", {1024}, 256},
            {"r64_d257", {64}, 257},
            {"r1024_d257", {1024}, 257},
            {"r64_d511", {64}, 511},
            {"r1024_d511", {1024}, 511},
            {"r64_d512", {64}, 512},
            {"r1024_d512", {1024}, 512},
            {"r64_d513", {64}, 513},
            {"r1024_d513", {1024}, 513},
            {"r64_d1023", {64}, 1023},
            {"r1024_d1023", {1024}, 1023},
            {"r64_d1024", {64}, 1024},
            {"r1024_d1024", {1024}, 1024},
            {"r64_d2047", {64}, 2047},
            {"r1024_d2047", {1024}, 2047},
            {"r64_d2048", {64}, 2048},
            {"r1024_d2048", {1024}, 2048},
            {"r64_d4095", {64}, 4095},
            {"r1024_d4095", {1024}, 4095},
            {"r64_d4096", {64}, 4096},
            {"r1024_d4096", {1024}, 4096},
            {"r64_d4097", {64}, 4097},
            {"r1024_d4097", {1024}, 4097},
            {"r64_d8192", {64}, 8192},
            {"r1024_d8192", {1024}, 8192},
            {"r64_d8193", {64}, 8193},
            {"r1024_d8193", {1024}, 8193},
            {"r64_d65536", {64}, 65536},
            {"r64_d65537", {64}, 65537},
            {"r1024_d65536", {1024}, 65536},
            {"r1024_d65537", {1024}, 65537},

            // Multi-axis contiguous reductions must retain the same dense [outer,reduction,inner] performance.
            {"r16x16_d33", {16, 16}, 33},
            {"r16x16_d128", {16, 16}, 128},
            {"r16x16_d256", {16, 16}, 256},
            {"r16x16_d257", {16, 16}, 257},
            {"r16x16_d511", {16, 16}, 511},
            {"r16x16_d512", {16, 16}, 512},
            {"r16x16_d513", {16, 16}, 513},
            {"r16x16_d1024", {16, 16}, 1024},
            {"r16x16_d2048", {16, 16}, 2048},
            {"r16x16_d4096", {16, 16}, 4096},
            {"r16x16_d4097", {16, 16}, 4097},
            {"r16x16_d8192", {16, 16}, 8192},
            {"r16x16_d8193", {16, 16}, 8193},
            {"r16x16_d65536", {16, 16}, 65536},
        };
        std::vector<DataType> dtypes = {
            DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_64BIT_TYPES
        dtypes.push_back(DataType::FP64);
#endif
        const std::vector<CubReductionOp> operations = {
            CubReductionOp::Sum,
            CubReductionOp::Mean,
            CubReductionOp::Min,
            CubReductionOp::Max,
            CubReductionOp::Product,
            CubReductionOp::L1Norm,
            CubReductionOp::L2Norm,
        };
        const std::vector<CubArgReductionOp> arg_operations = {
            CubArgReductionOp::ArgMin,
            CubArgReductionOp::ArgMax,
        };

        TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, device);
        Stream stream(device);
        for (const ReductionShape& shape : shapes) {
            if (focused_arg_x4 && !isFocusedArgX4Shape(shape.name)) {
                continue;
            }
            if (focused_arg_x4_awkward && !isFocusedArgX4AwkwardShape(shape.name)) {
                continue;
            }
            for (DataType dtype : dtypes) {
                if ((focused_arg_x4 || focused_arg_x4_awkward) && !isFocusedArgX4DType(dtype)) {
                    continue;
                }
                if (!focused_arg_x4 && !focused_arg_x4_awkward) {
                    for (CubReductionOp op : operations) {
                        runCase(shape, dtype, op, target_input_bytes, stream, gpu_placement);
                    }
                }
                for (CubArgReductionOp op : arg_operations) {
                    if ((focused_arg_x4 || focused_arg_x4_awkward) && op != CubArgReductionOp::ArgMin) {
                        continue;
                    }
                    runArgCase(shape, dtype, op, target_input_bytes, stream, gpu_placement);
                }
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CubReductionBenchmark failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
