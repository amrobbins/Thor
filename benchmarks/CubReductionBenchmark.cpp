#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "benchmarks/CubReductionBenchmarkCandidate.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionBenchmarking;

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

struct ExactReductionCase {
    const char* family;
    const char* name;
    std::vector<uint64_t> dimensions;
    std::vector<uint32_t> axes;
    CubReductionPath expected_production_path;
};

struct ArgCensusCase {
    const char* family;
    const char* name;
    std::vector<uint64_t> dimensions;
    std::vector<uint32_t> axes;
    CubReductionPath expected_production_path;
    bool benchmark_composed_dense = false;
};

struct ViewCensusCase {
    const char* family;
    const char* name;
    std::vector<uint64_t> dimensions;
    std::vector<uint64_t> strides;
    std::vector<uint32_t> axes;
    CubReductionPath expected_production_path;
    bool expect_dense_physical_permutation = false;
};

struct ExactTiming {
    double best_ms = 0.0;
    double median_ms = 0.0;
    double worst_ms = 0.0;
};

struct BenchmarkOptions {
    bool focused_arg_x4 = false;
    bool focused_arg_x4_awkward = false;
    bool reduction_census = false;
    bool arg_census = false;
    bool view_census = false;
    bool dense_run_stage_census = false;
    bool dense_stage_cost_calibration = false;
    bool list_reduction_candidates = false;
    std::vector<std::string> candidate_names;
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

std::string formatDimensions(const std::vector<uint64_t>& dimensions) {
    std::string formatted;
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) {
            formatted.push_back('x');
        }
        formatted += std::to_string(dimensions[i]);
    }
    return formatted;
}

std::string formatAxes(const std::vector<uint32_t>& axes) {
    std::string formatted;
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i != 0) {
            formatted.push_back(';');
        }
        formatted += std::to_string(axes[i]);
    }
    return formatted;
}

std::string formatStrides(const std::vector<uint64_t>& strides) {
    std::string formatted;
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i != 0) {
            formatted.push_back(';');
        }
        formatted += std::to_string(strides[i]);
    }
    return formatted;
}

uint64_t viewStorageSpanElements(const std::vector<uint64_t>& dimensions,
                                 const std::vector<uint64_t>& strides) {
    if (dimensions.size() != strides.size() || dimensions.empty()) {
        throw std::invalid_argument("View census requires equal non-empty dimension/stride ranks.");
    }
    uint64_t max_offset = 0;
    for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] == 0) {
            throw std::invalid_argument("View census does not support zero-sized dimensions.");
        }
        const uint64_t contribution = checkedMultiply(dimensions[axis] - 1, strides[axis], "view storage span");
        if (max_offset > std::numeric_limits<uint64_t>::max() - contribution) {
            throw std::overflow_error("View census storage span overflowed uint64_t.");
        }
        max_offset += contribution;
    }
    if (max_offset == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("View census storage span overflowed uint64_t.");
    }
    return max_offset + 1;
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
        case CubReductionPath::ComposedDense:
            return "composed_dense";
    }
    return "unknown";
}

int nextPowerOfTwoWarps(uint64_t required) {
    int warps = 1;
    while (static_cast<uint64_t>(warps) < CubReductionTiledPolicy::FULL_ROW_MAX_WARPS_PER_OUTPUT
           && static_cast<uint64_t>(warps) < required) {
        warps *= 2;
    }
    return static_cast<uint64_t>(warps) >= required ? warps : 0;
}

int groupedFullRowWarps(uint64_t inner_size, DataType input_dtype) {
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));
    const uint64_t component_warps = ceilDiv(inner_size, CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_WARP);
    const uint64_t stage_warps = ceilDiv(checkedMultiply(inner_size, element_bytes, "benchmark row bytes"),
                                         CubReductionTiledPolicy::ASYNC_STAGE_BYTES_PER_WARP);
    return nextPowerOfTwoWarps(std::max(component_warps, stage_warps));
}

const char* tiledStrategyName(uint64_t inner_size, DataType input_dtype) {
    if (inner_size == 32) {
        return "vector_direct_x1";
    }
    if (inner_size < 32) {
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
    if (inner_size == CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_WARP) {
        return "vector_direct_x16";
    }
    if (inner_size == CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_WARP * 2) {
        return "vector_direct_group2_x16";
    }
    if (inner_size == CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_WARP * 4) {
        return "vector_direct_group4_x16";
    }
    if (inner_size == CubReductionTiledPolicy::FULL_ROW_GROUP_MAX_INNER_SIZE) {
        return "vector_direct_group8_x16";
    }
    if (inner_size < CubReductionTiledPolicy::FULL_ROW_GROUP_MAX_INNER_SIZE) {
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
    if (inner_size > CubReductionTiledPolicy::FULL_ROW_GROUP_MAX_INNER_SIZE) {
        return inner_size % CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_BLOCK == 0
                   ? "vector_direct_block_shards_x16"
                   : "alignment_safe_vectorized_shaped_block_shards_x16";
    }
    return "direct_component_tiled";
}

std::string strategyName(const StampedCubReduction& reduction) {
    if (reduction.getPath() == CubReductionPath::TiledFixedSegment) {
        return tiledStrategyName(reduction.getGeometry().inner_size, reduction.getInputDataType());
    }
    if (reduction.getPath() == CubReductionPath::ComposedDense) {
        std::vector<std::vector<uint32_t>> stage_axes = reduction.getComposedStageAxes();
        if (stage_axes.empty()) {
            return "interval_dp_missing_stages";
        }
        if (stage_axes.size() == 1) {
            return "interval_dp_single_fp32";
        }
        std::vector<std::vector<uint32_t>> remaining = stage_axes;
        std::sort(remaining.begin(), remaining.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.front() < rhs.front();
        });
        std::string strategy = "interval_dp_";
        for (size_t stage_index = 0; stage_index + 1 < stage_axes.size(); ++stage_index) {
            const uint32_t selected_first_axis = stage_axes[stage_index].front();
            if (selected_first_axis == remaining.front().front()) {
                strategy.push_back('l');
                remaining.erase(remaining.begin());
            } else if (selected_first_axis == remaining.back().front()) {
                strategy.push_back('r');
                remaining.pop_back();
            } else {
                strategy.push_back('x');
                break;
            }
        }
        strategy += "_fp32";
        return strategy;
    }
    return pathName(reduction.getPath());
}

std::string tiledArgStrategyName(uint64_t outer_size,
                                 uint64_t inner_size,
                                 uint64_t reduction_size,
                                 DataType input_dtype,
                                 bool has_carried_index = false) {
    (void)outer_size;
    if (reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return "legacy_uint64_tiled";
    }
    if (inner_size <= 16) {
        int row_lanes = 1;
        if (inner_size <= 2) {
            row_lanes = 16;
        } else if (inner_size <= 4) {
            row_lanes = 8;
        } else if (inner_size <= 8) {
            row_lanes = 4;
        } else if (inner_size <= 16) {
            row_lanes = 2;
        }
        return "aligned_async_narrow_shared_rl" + std::to_string(row_lanes) + "_ordered_direct";
    }
    if (inner_size <= 32) {
        if (!has_carried_index) {
            return "aligned_async_narrow_cta2_shared_ordered_direct";
        }
        return "aligned_async_narrow_shared_rl1_carried";
    }

    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));
    if (element_bytes == 1) {
        return "legacy_fp8_tiled";
    }

    const uint64_t items_per_lane = 16 / element_bytes;
    int target_partitions = 1;
    while (target_partitions < 32 && static_cast<uint64_t>(target_partitions * 16) < reduction_size) {
        target_partitions <<= 1;
    }

    auto period_for_bytes = [&](uint64_t element_size) {
        int period = 1;
        const uint64_t row_bytes = inner_size * element_size;
        while (period < 16 && (row_bytes * static_cast<uint64_t>(period)) % 16 != 0) {
            period <<= 1;
        }
        return period;
    };
    int alignment_period = period_for_bytes(element_bytes);
    if (has_carried_index) {
        alignment_period = std::max(alignment_period, period_for_bytes(4));
    }
    const int row_partitions = std::max(target_partitions, alignment_period);

    // Match ARG-DIRECT-7: incomplete packets are edge work and must not promote the bulk packet geometry.
    const uint64_t complete_packets = inner_size / items_per_lane;
    int desired_group_threads = 8;
    while (desired_group_threads < 32
           && static_cast<uint64_t>(desired_group_threads << 1) <= complete_packets) {
        desired_group_threads <<= 1;
    }
    const int max_group_threads = std::min(32, 256 / row_partitions);
    const int group_threads = std::min(desired_group_threads, max_group_threads);

    const uint64_t tile_components = static_cast<uint64_t>(group_threads) * items_per_lane;
    const uint64_t full_tiles = inner_size / tile_components;
    const uint64_t remainder = inner_size % tile_components;
    const bool fold_small_tail = full_tiles != 0 && remainder != 0 && remainder <= items_per_lane;
    const bool value_rows_aligned = (inner_size * element_bytes) % 16 == 0;
    const bool carried_rows_aligned = !has_carried_index || (inner_size * uint64_t{4}) % 16 == 0;
    const bool simple_aligned_full_tiles = value_rows_aligned && carried_rows_aligned && remainder == 0;
    return "logical_group" + std::to_string(group_threads) + "_shared_rp" + std::to_string(row_partitions)
           + "_x" + std::to_string(items_per_lane) + "_16B_b"
           + std::to_string(row_partitions * group_threads)
           + (simple_aligned_full_tiles ? "_simple_aligned_full_tiles" : "_head_bulk_tail")
           + (alignment_period > 1 ? "_stride_phase" + std::to_string(alignment_period) : "_aligned_stride")
           + (!simple_aligned_full_tiles
                  ? (fold_small_tail ? "_folded_tail" : (remainder != 0 ? "_tail_tile" : ""))
                  : "");
}

std::string contiguousArgStrategyName(DataType input_dtype, uint64_t reduction_size) {
    if (reduction_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return "contiguous_segment_uint64_cub";
    }
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype));
    const uint64_t items_per_lane = 16 / element_bytes;
    const uint64_t packets = reduction_size / items_per_lane;
    int group_threads = element_bytes == 1 ? 32 : 1;
    while (group_threads < 32 && static_cast<uint64_t>(group_threads) < packets) {
        group_threads <<= 1;
    }
    int alignment_period = 1;
    const uint64_t segment_bytes = reduction_size * element_bytes;
    while (alignment_period < 16
           && (segment_bytes * static_cast<uint64_t>(alignment_period)) % 16 != 0) {
        alignment_period <<= 1;
    }
    return "grouped_contiguous_shared_g" + std::to_string(group_threads) + "_x"
           + std::to_string(items_per_lane) + "_16B_phase" + std::to_string(alignment_period);
}

std::string strategyName(const StampedCubArgReduction& reduction) {
    if (reduction.getPath() == CubReductionPath::TiledFixedSegment) {
        return tiledArgStrategyName(reduction.getGeometry().outer_size,
                                    reduction.getGeometry().inner_size,
                                    reduction.getGeometry().reduction_size,
                                    reduction.getInputDataType());
    }
    if (reduction.getPath() == CubReductionPath::ContiguousFixedSegment) {
        return contiguousArgStrategyName(reduction.getInputDataType(), reduction.getGeometry().reduction_size);
    }
    if (reduction.getPath() == CubReductionPath::DeviceTransformReduce
        && reduction.getGeometry().input_elements <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return "device_transform_compact_u32_cub";
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

template <typename RunFn>
ExactTiming timeExactReduction(Tensor& cache_flush, Stream& stream, RunFn&& run) {
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        run();
    }
    stream.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::vector<double> sample_ms;
    sample_ms.reserve(TIMING_SAMPLES);
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        // Exact-shape census cases preserve their production dimensions. Evict the previous working set between
        // samples instead of inflating a geometry merely to outrun L2. Eviction is completely outside timing.
        checkCuda(cudaMemsetAsync(cache_flush.getMemPtr<void>(),
                                  0x5a + sample,
                                  cache_flush.getArraySizeInBytes(),
                                  stream.getStream()),
                  "cudaMemsetAsync(cache_flush)");
        stream.synchronize();

        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(start)");
        run();
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        sample_ms.push_back(static_cast<double>(elapsed_ms));
    }
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::sort(sample_ms.begin(), sample_ms.end());
    return ExactTiming{sample_ms.front(), sample_ms[sample_ms.size() / 2], sample_ms.back()};
}

void printExactResult(const ExactReductionCase& benchmark_case,
                      DataType dtype,
                      CubReductionOp op,
                      std::string_view executor,
                      std::string_view implementation,
                      std::string_view strategy,
                      std::string_view index_bits,
                      uint64_t output_elements,
                      uint64_t reduction_elements_per_output,
                      std::optional<uint32_t> vector_elements_per_load,
                      std::optional<uint32_t> block_threads,
                      std::optional<uint64_t> first_stage_blocks,
                      std::optional<uint64_t> shards_per_output,
                      size_t scratch_bytes,
                      uint64_t input_bytes,
                      uint64_t output_bytes,
                      const ExactTiming& timing) {
    const uint64_t logical_bytes = input_bytes + output_bytes;
    const double logical_gb_per_second = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    auto printOptional = [](auto value) {
        if (value.has_value()) {
            std::cout << value.value();
        } else {
            std::cout << "backend_managed";
        }
    };

    std::cout << executor << ',' << benchmark_case.family << ',' << benchmark_case.name << ','
              << formatDimensions(benchmark_case.dimensions) << ',' << formatAxes(benchmark_case.axes) << ','
              << dataTypeName(dtype) << ',' << operationName(op) << ',' << implementation << ',' << strategy << ','
              << index_bits << ',' << output_elements << ',' << reduction_elements_per_output << ',';
    printOptional(vector_elements_per_load);
    std::cout << ',';
    printOptional(block_threads);
    std::cout << ',';
    printOptional(first_stage_blocks);
    std::cout << ',';
    printOptional(shards_per_output);
    std::cout << ',' << scratch_bytes << ',' << input_bytes << ',' << output_bytes << ',' << std::fixed
              << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
              << std::setprecision(2) << logical_gb_per_second << '\n';
}

void runProductionExactCase(const ExactReductionCase& benchmark_case,
                            DataType dtype,
                            CubReductionOp op,
                            Tensor& input,
                            Tensor& cache_flush,
                            Stream& stream) {
    std::shared_ptr<StampedCubReduction> stamped = CubReduction(op, benchmark_case.axes).stamp(input, stream);
    if (stamped->getPath() != benchmark_case.expected_production_path) {
        throw std::runtime_error(std::string("Reduction census path drift for case '") + benchmark_case.name
                                 + "': expected " + pathName(benchmark_case.expected_production_path) + ", got "
                                 + pathName(stamped->getPath()) + ". Update the census intentionally if production "
                                   "selection changed.");
    }

    const ExactTiming timing = timeExactReduction(cache_flush, stream, [&] { stamped->runOn(stream); });
    const Tensor output = stamped->getOutputTensor();
    const char* index_bits = stamped->getPath() == CubReductionPath::StridedFixedSegment
                                 ? (stamped->getGeometry().strided_value_indexing_fits_uint32 ? "32" : "64")
                                 : "n/a";

    printExactResult(benchmark_case,
                     dtype,
                     op,
                     "production",
                     pathName(stamped->getPath()),
                     strategyName(*stamped),
                     index_bits,
                     stamped->getGeometry().output_elements,
                     stamped->getGeometry().reduction_size,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     stamped->getWorkspaceSizeInBytes(),
                     input.getArraySizeInBytes(),
                     output.getArraySizeInBytes(),
                     timing);
}

float hostReductionValueAsFloat(const Tensor& tensor, uint64_t index) {
    switch (tensor.getDataType()) {
        case DataType::FP16:
            return static_cast<float>(tensor.getMemPtr<__half>()[index]);
        case DataType::BF16:
            return static_cast<float>(tensor.getMemPtr<__nv_bfloat16>()[index]);
        case DataType::FP32:
            return tensor.getMemPtr<float>()[index];
        default:
            throw std::logic_error("Reduction candidate validation supports FP16/BF16/FP32 outputs only.");
    }
}

void validateCandidateAgainstProduction(const ExactReductionCase& benchmark_case,
                                        CubReductionOp op,
                                        StampedReductionCandidate& candidate,
                                        Tensor& input,
                                        Stream& stream) {
    std::shared_ptr<StampedCubReduction> production = CubReduction(op, benchmark_case.axes).stamp(input, stream);
    production->runOn(stream);
    candidate.runOn(stream);
    stream.synchronize();

    const Tensor& production_gpu = production->getOutputTensor();
    const Tensor& candidate_gpu = candidate.getOutputTensor();
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    Tensor production_cpu(cpu_placement, production_gpu.getDescriptor());
    Tensor candidate_cpu(cpu_placement, candidate_gpu.getDescriptor());
    production_cpu.copyFromAsync(production_gpu, stream);
    candidate_cpu.copyFromAsync(candidate_gpu, stream);
    stream.synchronize();

    const bool low_precision = candidate_cpu.getDataType() == DataType::FP16
                               || candidate_cpu.getDataType() == DataType::BF16;
    const float absolute_tolerance = low_precision ? 0.5f : 0.05f;
    const float relative_tolerance = low_precision ? 0.02f : 0.001f;
    for (uint64_t i = 0; i < candidate_cpu.getTotalNumElements(); ++i) {
        const float expected = hostReductionValueAsFloat(production_cpu, i);
        const float actual = hostReductionValueAsFloat(candidate_cpu, i);
        const float tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
            throw std::runtime_error(std::string("Reduction benchmark candidate validation failed for case '")
                                     + benchmark_case.name + "' at output " + std::to_string(i) + ": expected "
                                     + std::to_string(expected) + ", got " + std::to_string(actual)
                                     + ", tolerance " + std::to_string(tolerance) + ".");
        }
    }
}

void runCandidateExactCase(const ExactReductionCase& benchmark_case,
                           DataType dtype,
                           CubReductionOp op,
                           const ReductionCandidate& candidate,
                           Tensor& input,
                           Tensor& cache_flush,
                           Stream& stream) {
    if (!candidate.supports(op, input.getDescriptor(), benchmark_case.axes)) {
        return;
    }

    std::unique_ptr<StampedReductionCandidate> stamped = candidate.stamp(op, input, benchmark_case.axes, stream);
    if (!stamped) {
        throw std::runtime_error(std::string("Reduction benchmark candidate '") + std::string(candidate.getName())
                                 + "' returned a null stamped candidate.");
    }
    const CandidateLaunchMetadata metadata = stamped->getLaunchMetadata();
    if (metadata.implementation.empty() || metadata.strategy.empty()) {
        throw std::runtime_error(std::string("Reduction benchmark candidate '") + std::string(candidate.getName())
                                 + "' must report non-empty implementation and strategy names.");
    }

    const ExactTiming timing = timeExactReduction(cache_flush, stream, [&] { stamped->runOn(stream); });
    const Tensor& output = stamped->getOutputTensor();
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(benchmark_case.dimensions, benchmark_case.axes);
    if (output.getDataType() != dtype || output.getDimensions() != geometry.output_dimensions) {
        throw std::runtime_error(std::string("Reduction benchmark candidate '") + std::string(candidate.getName())
                                 + "' returned an output descriptor that does not match CubReduction semantics.");
    }
    validateCandidateAgainstProduction(benchmark_case, op, *stamped, input, stream);

    const std::string executor = "candidate:" + std::string(candidate.getName());
    const std::string index_bits = metadata.index_bits.has_value() ? std::to_string(metadata.index_bits.value()) : "n/a";

    printExactResult(benchmark_case,
                     dtype,
                     op,
                     executor,
                     metadata.implementation,
                     metadata.strategy,
                     index_bits,
                     geometry.output_elements,
                     geometry.reduction_size,
                     metadata.vector_elements_per_load,
                     metadata.block_threads,
                     metadata.first_stage_blocks,
                     metadata.shards_per_output,
                     stamped->getWorkspaceSizeInBytes(),
                     input.getArraySizeInBytes(),
                     output.getArraySizeInBytes(),
                     timing);
}

template <typename RunFn>
ExactTiming timeExactReductionHot(Stream& stream, RunFn&& run) {
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        run();
    }
    stream.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(stage_hot_start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stage_hot_stop)");

    std::vector<double> sample_ms;
    sample_ms.reserve(TIMING_SAMPLES);
    for (int sample = 0; sample < TIMING_SAMPLES; ++sample) {
        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(stage_hot_start)");
        run();
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stage_hot_stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stage_hot_stop)");

        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(stage_hot)");
        sample_ms.push_back(static_cast<double>(elapsed_ms));
    }
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(stage_hot_start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stage_hot_stop)");

    std::sort(sample_ms.begin(), sample_ms.end());
    return ExactTiming{sample_ms.front(), sample_ms[sample_ms.size() / 2], sample_ms.back()};
}

struct DenseStageReducedRunSpan {
    uint32_t first_axis = 0;
    uint32_t last_axis = 0;
};

std::vector<DenseStageReducedRunSpan> denseStageReducedRunSpans(const ExactReductionCase& benchmark_case) {
    const CubReductionGeometry geometry =
        CubReduction::analyzeGeometry(benchmark_case.dimensions, benchmark_case.axes);
    if (!geometry.dense_run_geometry.has_value()) {
        return {};
    }

    std::vector<bool> is_reduced(benchmark_case.dimensions.size(), false);
    for (uint32_t axis : benchmark_case.axes) {
        is_reduced[axis] = true;
    }

    std::vector<DenseStageReducedRunSpan> spans;
    bool previous_non_singleton_was_reduced = false;
    for (uint32_t axis = 0; axis < benchmark_case.dimensions.size(); ++axis) {
        if (benchmark_case.dimensions[axis] == 1) {
            continue;
        }
        if (is_reduced[axis]) {
            if (!previous_non_singleton_was_reduced) {
                spans.push_back(DenseStageReducedRunSpan{axis, axis});
            } else {
                spans.back().last_axis = axis;
            }
        }
        previous_non_singleton_was_reduced = is_reduced[axis];
    }
    return spans;
}

std::vector<uint32_t> denseStageAxes(const DenseStageReducedRunSpan& span) {
    std::vector<uint32_t> axes;
    axes.reserve(static_cast<size_t>(span.last_axis - span.first_axis) + 1U);
    for (uint32_t axis = span.first_axis; axis <= span.last_axis; ++axis) {
        axes.push_back(axis);
    }
    return axes;
}

std::vector<uint64_t> denseStageStateDimensions(const ExactReductionCase& benchmark_case,
                                                const std::vector<DenseStageReducedRunSpan>& spans,
                                                size_t state_left,
                                                size_t state_right) {
    std::vector<uint64_t> dimensions = benchmark_case.dimensions;
    for (size_t run_index = 0; run_index < spans.size(); ++run_index) {
        if (run_index >= state_left && run_index <= state_right) {
            continue;
        }
        for (uint32_t axis = spans[run_index].first_axis; axis <= spans[run_index].last_axis; ++axis) {
            dimensions[axis] = 1;
        }
    }
    return dimensions;
}

void printDenseStageTiming(const ExactReductionCase& benchmark_case,
                           size_t state_left,
                           size_t state_right,
                           char choice,
                           size_t selected_run,
                           const std::vector<uint32_t>& stage_axes,
                           const std::vector<uint64_t>& input_dimensions,
                           DataType input_dtype,
                           DataType output_dtype,
                           const StampedCubReduction& stamped,
                           std::string_view cache_state,
                           const ExactTiming& timing) {
    const Tensor& output = stamped.getOutputTensor();
    const uint64_t input_elements = stamped.getGeometry().input_elements;
    const uint64_t input_bytes = checkedMultiply(
        input_elements,
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype)),
        "dense stage census input bytes");
    const uint64_t output_bytes = output.getArraySizeInBytes();
    const double logical_gb_per_second = static_cast<double>(input_bytes + output_bytes) / (timing.median_ms * 1.0e6);

    std::cout << benchmark_case.name << ',' << formatDimensions(benchmark_case.dimensions) << ','
              << formatAxes(benchmark_case.axes) << ',' << state_left << ',' << state_right << ','
              << (state_right - state_left + 1) << ',' << choice << ',' << selected_run << ','
              << formatAxes(stage_axes) << ',' << formatDimensions(input_dimensions) << ','
              << dataTypeName(input_dtype) << ',' << dataTypeName(output_dtype) << ',' << cache_state << ','
              << pathName(stamped.getPath()) << ',' << strategyName(stamped) << ',' << stamped.getGeometry().outer_size
              << ',' << stamped.getGeometry().reduction_size << ',' << stamped.getGeometry().inner_size << ','
              << input_bytes << ',' << output_bytes << ',' << std::fixed << std::setprecision(4) << timing.median_ms
              << ',' << timing.best_ms << ',' << timing.worst_ms << ',' << std::setprecision(2)
              << logical_gb_per_second << '\n';
}

void runDenseStageTiming(const ExactReductionCase& benchmark_case,
                         const std::vector<DenseStageReducedRunSpan>& spans,
                         size_t state_left,
                         size_t state_right,
                         char choice,
                         size_t selected_run,
                         DataType input_dtype,
                         DataType output_dtype,
                         Tensor& cache_flush,
                         Stream& stream,
                         const TensorPlacement& gpu_placement) {
    const std::vector<uint64_t> state_dimensions =
        denseStageStateDimensions(benchmark_case, spans, state_left, state_right);
    const std::vector<uint32_t> stage_axes = denseStageAxes(spans[selected_run]);
    Tensor input(gpu_placement, TensorDescriptor(input_dtype, state_dimensions));
    checkCuda(cudaMemsetAsync(input.getMemPtr<void>(), 0, input.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(dense_stage_input)");
    stream.synchronize();

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, stage_axes, output_dtype).stamp(input, stream);
    if (stamped->getPath() != CubReductionPath::DeviceTransformReduce
        && stamped->getPath() != CubReductionPath::ContiguousFixedSegment
        && stamped->getPath() != CubReductionPath::TiledFixedSegment) {
        throw std::runtime_error(std::string("Dense stage census case '") + benchmark_case.name
                                 + "' did not resolve to a direct dense reducer.");
    }

    const ExactTiming cold = timeExactReduction(cache_flush, stream, [&] { stamped->runOn(stream); });
    printDenseStageTiming(benchmark_case,
                          state_left,
                          state_right,
                          choice,
                          selected_run,
                          stage_axes,
                          state_dimensions,
                          input_dtype,
                          output_dtype,
                          *stamped,
                          "cold",
                          cold);

    // A composed non-root stage consumes an intermediate written immediately by its predecessor. Repeated execution
    // without the >=8x-L2 eviction buffer provides the complementary L2-resident measurement when that intermediate
    // actually fits in cache; inputs larger than L2 naturally remain streaming even in this mode.
    const ExactTiming hot = timeExactReductionHot(stream, [&] { stamped->runOn(stream); });
    printDenseStageTiming(benchmark_case,
                          state_left,
                          state_right,
                          choice,
                          selected_run,
                          stage_axes,
                          state_dimensions,
                          input_dtype,
                          output_dtype,
                          *stamped,
                          "hot",
                          hot);
}

void runDenseRunStageCensus(const std::vector<ExactReductionCase>& exact_cases,
                            const std::vector<DataType>& exact_dtypes,
                            Tensor& cache_flush,
                            Stream& stream,
                            const TensorPlacement& gpu_placement) {
    std::cout << "# mode=dense_run_stage_census operation=sum root_cache_state=cold "
                 "subproblem_cache_states=cold|hot production_selector=unchanged\n";
    std::cout << "case,original_dimensions,original_axes,state_left_run,state_right_run,remaining_runs,choice,"
                 "selected_run,stage_axes,input_dimensions,input_dtype,output_dtype,cache_state,path,strategy,outer,"
                 "reduction,inner,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    for (const ExactReductionCase& benchmark_case : exact_cases) {
        const std::vector<DenseStageReducedRunSpan> spans = denseStageReducedRunSpans(benchmark_case);
        if (spans.size() < 2) {
            continue;
        }
        const size_t last_run = spans.size() - 1;

        // Root decisions are the only stages that read the caller's original dtype and the only ones that are
        // obligatorily DRAM-cold in the cache-cold census. Measure all three production input dtypes.
        for (DataType dtype : exact_dtypes) {
            runDenseStageTiming(benchmark_case,
                                spans,
                                0,
                                last_run,
                                'L',
                                0,
                                dtype,
                                DataType::FP32,
                                cache_flush,
                                stream,
                                gpu_placement);
            runDenseStageTiming(benchmark_case,
                                spans,
                                0,
                                last_run,
                                'R',
                                last_run,
                                dtype,
                                DataType::FP32,
                                cache_flush,
                                stream,
                                gpu_placement);
        }

        // Every non-root end-elimination state is an FP32 partial aggregate. Benchmark both possible next direct
        // stages once as FP32->FP32; these state costs are shared by FP16/BF16/FP32 callers.
        for (size_t remaining = 2; remaining < spans.size(); ++remaining) {
            for (size_t state_left = 0; state_left + remaining <= spans.size(); ++state_left) {
                const size_t state_right = state_left + remaining - 1;
                runDenseStageTiming(benchmark_case,
                                    spans,
                                    state_left,
                                    state_right,
                                    'L',
                                    state_left,
                                    DataType::FP32,
                                    DataType::FP32,
                                    cache_flush,
                                    stream,
                                    gpu_placement);
                runDenseStageTiming(benchmark_case,
                                    spans,
                                    state_left,
                                    state_right,
                                    'R',
                                    state_right,
                                    DataType::FP32,
                                    DataType::FP32,
                                    cache_flush,
                                    stream,
                                    gpu_placement);
            }
        }

        // The final remaining run performs the only FP32->public-dtype conversion. Measure it separately for each
        // public dtype so a later planner model does not hide conversion/final-store cost inside an unrelated stage.
        for (size_t run_index = 0; run_index < spans.size(); ++run_index) {
            for (DataType dtype : exact_dtypes) {
                runDenseStageTiming(benchmark_case,
                                    spans,
                                    run_index,
                                    run_index,
                                    'F',
                                    run_index,
                                    DataType::FP32,
                                    dtype,
                                    cache_flush,
                                    stream,
                                    gpu_placement);
            }
        }
    }
}


struct DenseStageCostCalibrationCase {
    const char* family;
    uint64_t reduction;
    uint64_t inner;
    uint64_t target_fp32_input_bytes;
};

void printDenseStageCostCalibrationTiming(const DenseStageCostCalibrationCase& calibration_case,
                                          DataType input_dtype,
                                          DataType output_dtype,
                                          const StampedCubReduction& stamped,
                                          std::string_view cache_state,
                                          const ExactTiming& timing) {
    const Tensor& output = stamped.getOutputTensor();
    const uint64_t input_bytes = checkedMultiply(
        stamped.getGeometry().input_elements,
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(input_dtype)),
        "dense stage cost calibration input bytes");
    const uint64_t output_bytes = output.getArraySizeInBytes();
    const double logical_gb_per_second = static_cast<double>(input_bytes + output_bytes) / (timing.median_ms * 1.0e6);

    std::cout << calibration_case.family << ',' << dataTypeName(input_dtype) << ',' << dataTypeName(output_dtype)
              << ',' << cache_state << ',' << pathName(stamped.getPath()) << ',' << strategyName(stamped) << ','
              << stamped.getGeometry().outer_size << ',' << stamped.getGeometry().reduction_size << ','
              << stamped.getGeometry().inner_size << ',' << input_bytes << ',' << output_bytes << ',' << std::fixed
              << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
              << std::setprecision(2) << logical_gb_per_second << '\n';
}

void runDenseStageCostCalibrationTiming(const DenseStageCostCalibrationCase& calibration_case,
                                        DataType input_dtype,
                                        DataType output_dtype,
                                        Tensor& cache_flush,
                                        Stream& stream,
                                        const TensorPlacement& gpu_placement) {
    const uint64_t target_elements = std::max<uint64_t>(
        1,
        ceilDiv(calibration_case.target_fp32_input_bytes,
                static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(DataType::FP32))));
    const uint64_t elements_per_outer = checkedMultiply(
        calibration_case.reduction, calibration_case.inner, "dense stage calibration elements per outer");
    const uint64_t outer = std::max<uint64_t>(1, ceilDiv(target_elements, elements_per_outer));

    std::vector<uint64_t> dimensions;
    if (calibration_case.inner == 1) {
        dimensions = {outer, calibration_case.reduction};
    } else {
        dimensions = {outer, calibration_case.reduction, calibration_case.inner};
    }
    Tensor input(gpu_placement, TensorDescriptor(input_dtype, dimensions));
    checkCuda(cudaMemsetAsync(input.getMemPtr<void>(), 0, input.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(dense_stage_cost_calibration_input)");
    stream.synchronize();

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, {1}, output_dtype).stamp(input, stream);
    const CubReductionPath expected_path =
        calibration_case.inner == 1 ? CubReductionPath::ContiguousFixedSegment : CubReductionPath::TiledFixedSegment;
    if (stamped->getPath() != expected_path) {
        throw std::runtime_error(std::string("Dense stage cost calibration unexpectedly selected '")
                                 + pathName(stamped->getPath()) + "' instead of '" + pathName(expected_path) + "'.");
    }

    const ExactTiming cold = timeExactReduction(cache_flush, stream, [&] { stamped->runOn(stream); });
    printDenseStageCostCalibrationTiming(
        calibration_case, input_dtype, output_dtype, *stamped, "cold", cold);
    const ExactTiming hot = timeExactReductionHot(stream, [&] { stamped->runOn(stream); });
    printDenseStageCostCalibrationTiming(
        calibration_case, input_dtype, output_dtype, *stamped, "hot", hot);
}

void runDenseStageCostCalibration(Tensor& cache_flush,
                                  Stream& stream,
                                  const TensorPlacement& gpu_placement) {
    constexpr uint64_t ONE_KIB = 1024ULL;
    constexpr uint64_t ONE_MIB = 1024ULL * ONE_KIB;
    constexpr uint64_t target_sizes[] = {64ULL * ONE_KIB, ONE_MIB, 16ULL * ONE_MIB, 128ULL * ONE_MIB, 512ULL * ONE_MIB};
    constexpr uint64_t reductions[] = {7, 31, 127, 512};
    // Representatives intentionally cover every tuned TiledFixedSegment ownership family plus both sides of the
    // 4096-component block-shard boundary. Exact vector widths are retained because they have different launch code
    // from the adjacent awkward widths.
    constexpr uint64_t inners[] = {
        13, 31, 47, 64, 65, 97, 128, 193, 256, 383, 512, 769, 1024, 1537, 2048, 2813, 4095, 4096,
        4097, 8193, 32769, 131071,
    };
    constexpr uint64_t contiguous_reductions[] = {3, 7, 13, 23, 31, 97, 512, 3025};

    std::cout << "# mode=dense_stage_cost_calibration operation=sum cache_states=cold|hot "
                 "production_selector=unchanged\n";
    std::cout << "family,input_dtype,output_dtype,cache_state,path,strategy,outer,reduction,inner,input_bytes,"
                 "output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    const DataType public_dtypes[] = {DataType::FP16, DataType::BF16, DataType::FP32};
    auto run_case = [&](const DenseStageCostCalibrationCase& calibration_case) {
        // Root-stage calibration: caller dtype -> FP32. Hot is emitted too because it is useful for validating the
        // cache model even though a real composition root is cold.
        for (DataType dtype : public_dtypes) {
            runDenseStageCostCalibrationTiming(
                calibration_case, dtype, DataType::FP32, cache_flush, stream, gpu_placement);
        }
        // Final-stage conversion calibration. FP32->FP32 was already emitted above.
        runDenseStageCostCalibrationTiming(
            calibration_case, DataType::FP32, DataType::FP16, cache_flush, stream, gpu_placement);
        runDenseStageCostCalibrationTiming(
            calibration_case, DataType::FP32, DataType::BF16, cache_flush, stream, gpu_placement);
    };

    for (uint64_t target_bytes : target_sizes) {
        for (uint64_t reduction : contiguous_reductions) {
            run_case(DenseStageCostCalibrationCase{"contiguous", reduction, 1, target_bytes});
        }
        for (uint64_t reduction : reductions) {
            for (uint64_t inner : inners) {
                run_case(DenseStageCostCalibrationCase{"tiled", reduction, inner, target_bytes});
            }
        }
    }
}

const ReductionCandidate* findReductionCandidate(std::string_view name) {
    for (const std::unique_ptr<ReductionCandidate>& candidate : getReductionCandidates()) {
        if (candidate->getName() == name) {
            return candidate.get();
        }
    }
    return nullptr;
}

std::string formatArgStageOrder(const CubArgReductionDenseCompositionPlan& plan) {
    std::string formatted;
    if (plan.topology.stages.empty()) {
        return formatted;
    }

    uint32_t left = 0;
    uint32_t right = static_cast<uint32_t>(plan.topology.stages.size() - 1);
    for (size_t stage_index = 0; stage_index < plan.topology.stages.size(); ++stage_index) {
        if (stage_index != 0) {
            formatted.push_back('>');
        }
        const uint32_t ordinal = plan.topology.stages[stage_index].reduced_run_ordinal;
        if (ordinal == left) {
            formatted.push_back('L');
            ++left;
        } else if (ordinal == right) {
            formatted.push_back('R');
            if (right != 0) {
                --right;
            }
        } else {
            formatted.push_back('X');
            formatted += std::to_string(ordinal);
        }
    }
    return formatted;
}

std::string formatArgStageAxes(const CubArgReductionDenseCompositionPlan& plan) {
    std::string formatted;
    for (size_t stage_index = 0; stage_index < plan.topology.stages.size(); ++stage_index) {
        if (stage_index != 0) {
            formatted.push_back('>');
        }
        const std::vector<uint32_t>& axes = plan.topology.stages[stage_index].reduction_axes;
        for (size_t axis_index = 0; axis_index < axes.size(); ++axis_index) {
            if (axis_index != 0) {
                formatted.push_back('+');
            }
            formatted += std::to_string(axes[axis_index]);
        }
    }
    return formatted;
}

std::string formatArgStagePaths(const CubArgReductionDenseCompositionPlan& plan) {
    std::string formatted;
    for (size_t stage_index = 0; stage_index < plan.topology.stages.size(); ++stage_index) {
        if (stage_index != 0) {
            formatted.push_back('>');
        }
        formatted += pathName(plan.topology.stages[stage_index].expected_path);
    }
    return formatted;
}

std::string formatArgStageStrategies(const CubArgReductionDenseCompositionPlan& plan, DataType original_input_dtype) {
    std::string formatted;
    for (size_t stage_index = 0; stage_index < plan.topology.stages.size(); ++stage_index) {
        if (stage_index != 0) {
            formatted.push_back('>');
        }
        const CubReductionDenseCompositionStage& stage = plan.topology.stages[stage_index];
        const DataType stage_input_dtype = stage_index == 0 ? original_input_dtype : DataType::FP32;
        if (stage.expected_path == CubReductionPath::TiledFixedSegment) {
            formatted += tiledArgStrategyName(stage.outer_size,
                                               stage.inner_size,
                                               stage.stage_reduction_size,
                                               stage_input_dtype,
                                               stage_index != 0);
        } else if (stage.expected_path == CubReductionPath::ContiguousFixedSegment) {
            formatted += contiguousArgStrategyName(stage_input_dtype, stage.stage_reduction_size);
        } else {
            formatted += pathName(stage.expected_path);
        }
    }
    return formatted;
}

std::string argProductionAccumulatorIndexBits(const StampedCubArgReduction& reduction) {
    switch (reduction.getPath()) {
        case CubReductionPath::TiledFixedSegment:
            return reduction.getGeometry().reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                       ? "32"
                       : "64";
        case CubReductionPath::DeviceTransformReduce:
            return reduction.getGeometry().input_elements <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                       ? "32"
                       : "64";
        case CubReductionPath::ContiguousFixedSegment:
            return reduction.getGeometry().reduction_size <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                       ? "32"
                       : "64";
        case CubReductionPath::StridedFixedSegment:
            // The irregular logical-index fallback still carries the legacy uint64_t candidate index.
            return "64";
        case CubReductionPath::OffsetSegmented:
        case CubReductionPath::ComposedDense:
            return "n/a";
    }
    return "n/a";
}

std::vector<std::vector<uint32_t>> enumerateArgEndEliminationRunOrders(uint32_t reduced_run_count) {
    if (reduced_run_count == 0) {
        return {};
    }
    if (reduced_run_count >= 64) {
        throw std::invalid_argument("ARG run-order census supports at most 63 reduced runs.");
    }

    const uint64_t decision_count = uint64_t{1} << (reduced_run_count - 1);
    std::vector<std::vector<uint32_t>> orders;
    orders.reserve(static_cast<size_t>(decision_count));
    for (uint64_t decisions = 0; decisions < decision_count; ++decisions) {
        uint32_t left = 0;
        uint32_t right = reduced_run_count - 1;
        std::vector<uint32_t> order;
        order.reserve(reduced_run_count);
        for (uint32_t step = 0; step + 1 < reduced_run_count; ++step) {
            const bool choose_right = ((decisions >> step) & uint64_t{1}) != 0;
            if (choose_right) {
                order.push_back(right--);
            } else {
                order.push_back(left++);
            }
        }
        THOR_THROW_IF_FALSE(left == right);
        order.push_back(left);
        orders.push_back(std::move(order));
    }
    return orders;
}

uint64_t argComposedIntermediateBytes(const CubArgReductionDenseCompositionPlan& plan) {
    if (plan.topology.stages.size() <= 1) {
        return 0;
    }
    const uint64_t index_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(plan.carried_index_dtype));
    const uint64_t bytes_per_candidate = sizeof(float) + index_bytes;
    uint64_t total_bytes = 0;
    for (size_t stage_index = 0; stage_index + 1 < plan.topology.stages.size(); ++stage_index) {
        const uint64_t stage_bytes = checkedMultiply(plan.topology.stages[stage_index].output_elements,
                                                     bytes_per_candidate,
                                                     "ARG census intermediate bytes");
        if (total_bytes > std::numeric_limits<uint64_t>::max() - stage_bytes) {
            throw std::overflow_error("ARG census intermediate bytes overflowed uint64_t.");
        }
        total_bytes += stage_bytes;
    }
    return total_bytes;
}

void validateArgIndexOutputs(const ArgCensusCase& benchmark_case,
                             CubArgReductionOp op,
                             const StampedCubArgReduction& production,
                             const StampedCubArgReduction& composed,
                             Stream& stream) {
    production.runOn(stream);
    composed.runOn(stream);
    stream.synchronize();

    if (!production.getIndexOutputTensor().has_value() || !composed.getIndexOutputTensor().has_value()) {
        throw std::logic_error("ARG census validation requires index-producing reductions.");
    }
    const Tensor& production_gpu = production.getIndexOutputTensor().value();
    const Tensor& composed_gpu = composed.getIndexOutputTensor().value();
    if (production_gpu.getDataType() != DataType::UINT32 || composed_gpu.getDataType() != DataType::UINT32
        || production_gpu.getDimensions() != composed_gpu.getDimensions()) {
        throw std::runtime_error(std::string("ARG census output descriptor mismatch for case '")
                                 + benchmark_case.name + "'.");
    }

    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    Tensor production_cpu(cpu_placement, production_gpu.getDescriptor());
    Tensor composed_cpu(cpu_placement, composed_gpu.getDescriptor());
    production_cpu.copyFromAsync(production_gpu, stream);
    composed_cpu.copyFromAsync(composed_gpu, stream);
    stream.synchronize();

    const uint32_t* expected = production_cpu.getMemPtr<uint32_t>();
    const uint32_t* actual = composed_cpu.getMemPtr<uint32_t>();
    for (uint64_t i = 0; i < production_cpu.getTotalNumElements(); ++i) {
        if (expected[i] != actual[i]) {
            throw std::runtime_error(std::string("ARG census composed validation failed for case '")
                                     + benchmark_case.name + "' operation " + operationName(op) + " at output "
                                     + std::to_string(i) + ": production index " + std::to_string(expected[i])
                                     + ", composed index " + std::to_string(actual[i]) + ".");
        }
    }
}

void printArgCensusResult(const ArgCensusCase& benchmark_case,
                          DataType dtype,
                          CubArgReductionOp op,
                          std::string_view executor,
                          std::string_view implementation,
                          std::string_view strategy,
                          std::string_view stage_order,
                          std::string_view stage_axes,
                          std::string_view stage_paths,
                          std::string_view index_bits,
                          size_t workspace_bytes,
                          uint64_t intermediate_bytes,
                          uint64_t input_bytes,
                          uint64_t output_bytes,
                          const ExactTiming& timing,
                          double speedup_vs_production) {
    const uint64_t logical_bytes = input_bytes + output_bytes;
    const double logical_gb_per_second = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    std::cout << executor << ',' << benchmark_case.family << ',' << benchmark_case.name << ','
              << formatDimensions(benchmark_case.dimensions) << ',' << formatAxes(benchmark_case.axes) << ','
              << dataTypeName(dtype) << ',' << operationName(op) << ',' << implementation << ',' << strategy << ','
              << stage_order << ',' << stage_axes << ',' << stage_paths << ',' << index_bits << ',' << workspace_bytes << ','
              << intermediate_bytes << ',' << input_bytes << ',' << output_bytes << ',' << std::fixed
              << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
              << std::setprecision(2) << logical_gb_per_second << ',' << std::setprecision(4)
              << speedup_vs_production << ",ok\n";
}

void runArgCensusCase(const ArgCensusCase& benchmark_case,
                      DataType dtype,
                      CubArgReductionOp op,
                      Tensor& cache_flush,
                      Stream& stream,
                      const TensorPlacement& gpu_placement) {
    Tensor input(gpu_placement, TensorDescriptor(dtype, benchmark_case.dimensions));
    input.fillRandom(-100.0, 100.0, stream);
    stream.synchronize();

    CubArgReductionOutputOptions outputs;
    outputs.produce_value = false;
    outputs.produce_index = true;
    outputs.index_output_dtype = DataType::UINT32;
    CubArgReduction reduction(op, benchmark_case.axes, outputs);
    std::shared_ptr<StampedCubArgReduction> production = reduction.stamp(input, stream);
    std::optional<CubArgReductionDenseCompositionPlan> production_plan;
    if (production->getPath() == CubReductionPath::ComposedDense) {
        production_plan = CubArgReduction::analyzeDenseCompositionPlanForExecution(
            benchmark_case.dimensions, benchmark_case.axes, dtype, outputs, stream);
        if (!production_plan.has_value()) {
            throw std::runtime_error(std::string("ARG census production ComposedDense plan missing for case '")
                                     + benchmark_case.name + "'.");
        }
    }
    if (production->getPath() != benchmark_case.expected_production_path) {
        throw std::runtime_error(std::string("ARG census path drift for case '") + benchmark_case.name
                                 + "': expected " + pathName(benchmark_case.expected_production_path) + ", got "
                                 + pathName(production->getPath()) + ". Update the census intentionally if production "
                                   "selection changed.");
    }

    uint32_t forced_reduced_run_count = 0;
    if (benchmark_case.benchmark_composed_dense) {
        const std::optional<CubArgReductionDenseCompositionPlan> structural_plan =
            CubArgReduction::analyzeDenseCompositionPlan(benchmark_case.dimensions, benchmark_case.axes);
        if (!structural_plan.has_value()) {
            throw std::runtime_error(std::string("ARG census expected a dense composition plan for case '")
                                     + benchmark_case.name + "'.");
        }
        forced_reduced_run_count = static_cast<uint32_t>(structural_plan->topology.stages.size());
    }

    const ExactTiming production_timing =
        timeExactReduction(cache_flush, stream, [&] { production->runOn(stream); });
    const uint64_t production_output_bytes = production->getIndexOutputTensor()->getArraySizeInBytes();
    const std::string production_index_bits = production_plan.has_value()
                                                  ? (production_plan->carried_index_dtype == DataType::UINT32 ? "32" : "64")
                                                  : argProductionAccumulatorIndexBits(*production);
    const std::string production_strategy = production_plan.has_value()
                                                ? formatArgStageStrategies(production_plan.value(), dtype)
                                                : strategyName(*production);
    const std::string production_stage_order = production_plan.has_value()
                                                   ? formatArgStageOrder(production_plan.value())
                                                   : "direct";
    const std::string production_stage_axes = production_plan.has_value()
                                                  ? formatArgStageAxes(production_plan.value())
                                                  : formatAxes(benchmark_case.axes);
    const std::string production_stage_paths = production_plan.has_value()
                                                   ? formatArgStagePaths(production_plan.value())
                                                   : pathName(production->getPath());
    const uint64_t production_intermediate_bytes = production_plan.has_value()
                                                       ? argComposedIntermediateBytes(production_plan.value())
                                                       : 0;
    printArgCensusResult(benchmark_case,
                         dtype,
                         op,
                         "production",
                         pathName(production->getPath()),
                         production_strategy,
                         production_stage_order,
                         production_stage_axes,
                         production_stage_paths,
                         production_index_bits,
                         production->getWorkspaceSizeInBytes(),
                         production_intermediate_bytes,
                         input.getArraySizeInBytes(),
                         production_output_bytes,
                         production_timing,
                         1.0);

    if (forced_reduced_run_count != 0) {
        for (const std::vector<uint32_t>& run_order :
             enumerateArgEndEliminationRunOrders(forced_reduced_run_count)) {
            const std::optional<CubArgReductionDenseCompositionPlan> forced_plan =
                CubArgReduction::analyzeDenseCompositionPlanForRunOrder(
                    benchmark_case.dimensions, benchmark_case.axes, run_order);
            if (!forced_plan.has_value()) {
                throw std::runtime_error(std::string("ARG census could not stamp legal end-elimination order for case '")
                                         + benchmark_case.name + "'.");
            }
            std::shared_ptr<StampedCubArgReduction> forced =
                reduction.stampComposedDenseWithPlan(input, forced_plan.value(), stream);
            if (forced->getPath() != CubReductionPath::ComposedDense) {
                throw std::runtime_error(std::string("ARG census forced-order executor did not select ComposedDense for case '")
                                         + benchmark_case.name + "'.");
            }
            validateArgIndexOutputs(benchmark_case, op, *production, *forced, stream);
            const ExactTiming forced_timing =
                timeExactReduction(cache_flush, stream, [&] { forced->runOn(stream); });
            const uint64_t forced_output_bytes = forced->getIndexOutputTensor()->getArraySizeInBytes();
            const double speedup = production_timing.median_ms / forced_timing.median_ms;
            const CubArgReductionDenseCompositionPlan& plan = forced_plan.value();
            const std::string index_bits = plan.carried_index_dtype == DataType::UINT32 ? "32" : "64";
            printArgCensusResult(benchmark_case,
                                 dtype,
                                 op,
                                 "forced_order",
                                 "composed_dense",
                                 formatArgStageStrategies(plan, dtype),
                                 formatArgStageOrder(plan),
                                 formatArgStageAxes(plan),
                                 formatArgStagePaths(plan),
                                 index_bits,
                                 forced->getWorkspaceSizeInBytes(),
                                 argComposedIntermediateBytes(plan),
                                 input.getArraySizeInBytes(),
                                 forced_output_bytes,
                                 forced_timing,
                                 speedup);
        }
    }
}

void runArgCensus(Tensor& cache_flush, Stream& stream, const TensorPlacement& gpu_placement) {
    // Keep this list intentionally explicit. It is a performance contract over each production dense ARG family and
    // every composition topology that ARG-PLAN is expected to replace. Boundary cases flank each Tiled ownership or
    // vectorization transition so cliffs remain visible instead of being averaged away by a broad random sweep.
    const std::vector<ArgCensusCase> cases = {
        {"direct_device", "full_r16777216", {16777216}, {0}, CubReductionPath::DeviceTransformReduce, false},
        {"direct_contiguous", "suffix_r64", {262144, 64}, {1}, CubReductionPath::ContiguousFixedSegment, false},
        {"direct_contiguous", "suffix_r65", {258111, 65}, {1}, CubReductionPath::ContiguousFixedSegment, false},
        {"direct_contiguous", "suffix_r128", {131072, 128}, {1}, CubReductionPath::ContiguousFixedSegment, false},
        {"direct_contiguous", "suffix_r256", {65536, 256}, {1}, CubReductionPath::ContiguousFixedSegment, false},
        {"direct_contiguous", "suffix_r1024", {16384, 1024}, {1}, CubReductionPath::ContiguousFixedSegment, false},
        {"direct_contiguous", "suffix_rr16x16", {65536, 16, 16}, {1, 2}, CubReductionPath::ContiguousFixedSegment, false},

        {"direct_tiled_row_lanes", "r256_i2", {32768, 256, 2}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i3", {21845, 256, 3}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i4", {16384, 256, 4}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i5", {13107, 256, 5}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i8", {8192, 256, 8}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i9", {7281, 256, 9}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i15", {4369, 256, 15}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i16", {4096, 256, 16}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i17", {3855, 256, 17}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i31", {2114, 256, 31}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i32", {2048, 256, 32}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i33", {1985, 256, 33}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_row_lanes", "r256_i63", {1040, 256, 63}, {1}, CubReductionPath::TiledFixedSegment, false},

        {"direct_tiled_vector", "r256_i64", {1024, 256, 64}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i65", {1008, 256, 65}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i128", {512, 256, 128}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i129", {508, 256, 129}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i256", {256, 256, 256}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i257", {255, 256, 257}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i511", {128, 256, 511}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i512", {128, 256, 512}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i513", {128, 256, 513}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i1024", {64, 256, 1024}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_vector", "r256_i1025", {64, 256, 1025}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i2048", {32, 256, 2048}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i4096", {16, 256, 4096}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i4097", {16, 256, 4097}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i8192", {8, 256, 8192}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i8193", {8, 256, 8193}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i65536", {1, 256, 65536}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_shards", "r256_i65537", {1, 256, 65537}, {1}, CubReductionPath::TiledFixedSegment, false},

        {"direct_tiled_reduction_sensitivity", "r64_i8", {32768, 64, 8}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r1024_i8", {2048, 1024, 8}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r64_i64", {4096, 64, 64}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r65_i64", {4032, 65, 64}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r128_i64", {2048, 128, 64}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r129_i64", {2032, 129, 64}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r64_i128", {2048, 64, 128}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r1024_i128", {128, 1024, 128}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r64_i4097", {64, 64, 4097}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_reduction_sensitivity", "r1024_i4097", {4, 1024, 4097}, {1}, CubReductionPath::TiledFixedSegment, false},
        {"direct_tiled_multi_axis", "rr16x16_i128", {512, 16, 16, 128}, {1, 2}, CubReductionPath::TiledFixedSegment, false},

        {"composed_two_runs", "rkr_trailing_reduced", {257, 63, 31}, {0, 2}, CubReductionPath::ComposedDense, true},
        {"composed_two_runs", "rkrk_trailing_retained", {31, 17, 29, 19}, {0, 2}, CubReductionPath::ComposedDense, true},
        {"composed_two_runs", "krkr_trailing_reduced", {17, 31, 19, 29}, {1, 3}, CubReductionPath::ComposedDense, true},
        {"composed_two_runs", "krkrk_trailing_retained", {17, 31, 19, 29, 23}, {1, 3}, CubReductionPath::ComposedDense, true},
        {"composed_three_runs", "rkrkr", {31, 17, 29, 19, 23}, {0, 2, 4}, CubReductionPath::ComposedDense, true},
        {"composed_three_runs", "krkrkr", {7, 31, 11, 29, 13, 23}, {1, 3, 5}, CubReductionPath::ComposedDense, true},
        {"composed_four_runs", "rkrkrkr", {13, 5, 11, 7, 9, 11, 7}, {0, 2, 4, 6}, CubReductionPath::ComposedDense, true},
        {"composed_four_runs", "krkrkrkr", {2, 13, 3, 11, 5, 9, 7, 7}, {1, 3, 5, 7}, CubReductionPath::ComposedDense, true},
        {"composed_multi_axis", "krrkrr", {17, 31, 7, 19, 29, 5}, {1, 2, 4, 5}, CubReductionPath::ComposedDense, true},
        {"composed_multi_axis", "rrkrrkrr", {13, 5, 11, 17, 3, 19, 7}, {0, 1, 3, 4, 6}, CubReductionPath::ComposedDense, true},
        {"composed_singleton", "r1rk_singleton_separator", {31, 1, 29, 19}, {0, 2}, CubReductionPath::ComposedDense, true},
        {"composed_rank9", "rank9_alternating", {5, 7, 3, 11, 2, 13, 3, 17, 5}, {1, 3, 5, 7}, CubReductionPath::ComposedDense, true},
    };

    const std::vector<DataType> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
    const std::vector<CubArgReductionOp> operations = {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax};

    std::cout << "# mode=arg_census operations=argmin_index|argmax_index dtypes=fp16|bf16|fp32 "
                 "cache_state=cold validation=production_exact_index_match production_selector=dense_disjoint_to_composed\n";
    std::cout << "# Direct families and cost-planned dense composition are measured through production stamp(); each representative dense disjoint-run case additionally forces "
                 "every legal left/right end-elimination order. speedup_vs_production > 1 means that forced order is faster than production.\n";
    std::cout << "# index_bits reports the hot candidate/carried-index width; the public benchmark output remains UINT32.\n";
    std::cout << "executor,family,case,dimensions,axes,dtype,operation,implementation,strategy,stage_order,stage_axes,stage_paths,"
                 "index_bits,workspace_bytes,intermediate_bytes,input_bytes,output_bytes,median_ms,best_ms,worst_ms,"
                 "logical_GBps,speedup_vs_production,validation\n";

    for (const ArgCensusCase& benchmark_case : cases) {
        for (DataType dtype : dtypes) {
            for (CubArgReductionOp op : operations) {
                runArgCensusCase(benchmark_case, dtype, op, cache_flush, stream, gpu_placement);
            }
        }
    }
}

void validateViewCensusZeroSum(const ViewCensusCase& benchmark_case,
                               StampedCubReduction& stamped,
                               Stream& stream) {
    stamped.runOn(stream);
    stream.synchronize();

    const Tensor& output_gpu = stamped.getOutputTensor();
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    Tensor output_cpu(cpu_placement, output_gpu.getDescriptor());
    output_cpu.copyFromAsync(output_gpu, stream);
    stream.synchronize();

    const uint64_t output_elements = output_cpu.getTotalNumElements();
    for (uint64_t i = 0; i < output_elements; ++i) {
        const float value = hostReductionValueAsFloat(output_cpu, i);
        if (value != 0.0f) {
            throw std::runtime_error(std::string("View census zero-SUM validation failed for case '")
                                     + benchmark_case.name + "' at output " + std::to_string(i) + ": got "
                                     + std::to_string(value) + ".");
        }
    }
}

void runViewCensusCase(const ViewCensusCase& benchmark_case,
                       DataType dtype,
                       Tensor& cache_flush,
                       Stream& stream,
                       const TensorPlacement& gpu_placement) {
    const CubReductionGeometry analyzed = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, benchmark_case.dimensions, benchmark_case.strides, benchmark_case.axes);
    if (analyzed.path != benchmark_case.expected_production_path) {
        throw std::runtime_error(std::string("View census path drift for case '") + benchmark_case.name
                                 + "': expected " + pathName(benchmark_case.expected_production_path) + ", got "
                                 + pathName(analyzed.path) + ". Update VIEW-1A intentionally if production selection changed.");
    }
    if (analyzed.physical_layout_is_dense_permutation != benchmark_case.expect_dense_physical_permutation) {
        throw std::runtime_error(std::string("View census dense-permutation classification drift for case '")
                                 + benchmark_case.name + "'.");
    }

    const uint64_t storage_elements = viewStorageSpanElements(benchmark_case.dimensions, benchmark_case.strides);
    Tensor storage(gpu_placement, TensorDescriptor(dtype, {storage_elements}));
    Tensor input = storage.aliasView(benchmark_case.dimensions, benchmark_case.strides);

    // Initialization and validation stay outside the timed interval. Zero input gives every logical view, including
    // overlapping and broadcast aliases, the same exact SUM reference: every output must be zero. VIEW-1A is a path
    // and performance census rather than a replacement-kernel correctness suite; VIEW-1 will add independent cursor
    // reference tests when execution changes.
    checkCuda(cudaMemsetAsync(storage.getMemPtr<void>(), 0, storage.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(view census storage)");
    stream.synchronize();

    std::shared_ptr<StampedCubReduction> stamped = CubReduction(CubReductionOp::Sum, benchmark_case.axes).stamp(input, stream);
    if (stamped->getPath() != benchmark_case.expected_production_path) {
        throw std::runtime_error(std::string("View census stamped path drift for case '") + benchmark_case.name
                                 + "': expected " + pathName(benchmark_case.expected_production_path) + ", got "
                                 + pathName(stamped->getPath()) + ".");
    }
    validateViewCensusZeroSum(benchmark_case, *stamped, stream);

    const ExactTiming timing = timeExactReduction(cache_flush, stream, [&] { stamped->runOn(stream); });
    const Tensor& output = stamped->getOutputTensor();
    const uint64_t element_bytes =
        static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
    const uint64_t logical_input_bytes = input.getArraySizeInBytes();
    const uint64_t storage_span_bytes = checkedMultiply(storage_elements, element_bytes, "view storage bytes");
    const uint64_t output_bytes = output.getArraySizeInBytes();
    const uint64_t logical_bytes = logical_input_bytes + output_bytes;
    const double logical_gb_per_second = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    const char* index_bits = stamped->getPath() == CubReductionPath::StridedFixedSegment
                                 ? (stamped->getGeometry().strided_value_indexing_fits_uint32 ? "32" : "64")
                                 : "n/a";

    std::cout << benchmark_case.family << ',' << benchmark_case.name << ','
              << formatDimensions(benchmark_case.dimensions) << ',' << formatStrides(benchmark_case.strides) << ','
              << formatAxes(benchmark_case.axes) << ',' << dataTypeName(dtype) << ",sum," << pathName(stamped->getPath())
              << ',' << strategyName(*stamped) << ','
              << (stamped->getGeometry().physical_layout_is_dense_permutation ? "true" : "false") << ','
              << (stamped->getGeometry().permutation_aware_tiled_geometry.has_value() ? "true" : "false") << ','
              << index_bits << ',' << stamped->getGeometry().rank << ',' << stamped->getGeometry().output_elements << ','
              << stamped->getGeometry().reduction_size << ',' << stamped->getWorkspaceSizeInBytes() << ','
              << logical_input_bytes << ',' << storage_span_bytes << ',' << output_bytes << ',' << std::fixed
              << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
              << std::setprecision(2) << logical_gb_per_second << ",ok\n";
}

void runViewCensus(Tensor& cache_flush, Stream& stream, const TensorPlacement& gpu_placement) {
    // VIEW-1A freezes the population that still depends on the legacy logical-index mapper before VIEW-1 replaces it.
    // The final three cases are deliberate controls proving that existing rank-1 affine, supported permutation, and
    // ordinary dense-composition paths remain outside the arbitrary-view fallback.
    const std::vector<ViewCensusCase> cases = {
        {"arbitrary_gapped", "gapped_reduction_stride", {4096, 127, 65}, {10160, 80, 1}, {1},
         CubReductionPath::StridedFixedSegment, false},
        {"arbitrary_gapped", "stride2_slice", {2048, 127, 65}, {16510, 130, 2}, {1},
         CubReductionPath::StridedFixedSegment, false},
        {"arbitrary_broadcast", "zero_stride_reduced_axis", {8192, 127, 65}, {0, 65, 1}, {0},
         CubReductionPath::StridedFixedSegment, false},
        {"arbitrary_overlap", "overlapping_outer_reduction", {4096, 127, 65}, {65, 65, 1}, {1},
         CubReductionPath::StridedFixedSegment, false},
        {"compact_permutation_fallback", "split_physical_reduction", {31, 29, 17, 23},
         {667, 23, 20677, 1}, {1, 2}, CubReductionPath::StridedFixedSegment, true},
        {"compact_permutation_fallback", "unsupported_retained_order", {31, 29, 17, 23},
         {667, 23, 20677, 1}, {0}, CubReductionPath::StridedFixedSegment, true},
        {"compact_permutation_fallback", "physically_trailing_reduction", {256, 257, 127},
         {1, 32512, 256}, {0}, CubReductionPath::StridedFixedSegment, true},
        {"compact_permutation_fallback", "full_reduction_rank3", {256, 257, 127},
         {1, 32512, 256}, {0, 1, 2}, CubReductionPath::StridedFixedSegment, true},
        {"arbitrary_singleton", "singleton_heavy_gapped", {2048, 1, 127, 1, 65},
         {10160, 10160, 80, 80, 1}, {0, 2}, CubReductionPath::StridedFixedSegment, false},
        // The singleton axis contributes no address span but its >UINT32_MAX stride forces the current mapper's
        // UINT64 metadata/index arithmetic, so VIEW-1 has a baseline for both legacy indexing widths.
        {"arbitrary_index_width", "uint64_indexing_singleton_stride", {2048, 1, 127, 65},
         {10160, (1ULL << 32), 80, 1}, {0, 2}, CubReductionPath::StridedFixedSegment, false},

        {"control_direct", "rank1_affine_stride2", {16777216}, {2}, {0},
         CubReductionPath::DeviceTransformReduce, false},
        {"control_direct", "supported_dense_permutation", {256, 257, 127}, {1, 32512, 256}, {2},
         CubReductionPath::TiledFixedSegment, true},
        {"control_dense", "ordinary_dense_disjoint", {31, 17, 29, 19, 23},
         {215441, 12673, 437, 23, 1}, {0, 2, 4}, CubReductionPath::ComposedDense, true},
    };

    const std::vector<DataType> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};

    std::cout << "# mode=view_census operation=sum dtypes=fp16|bf16|fp32 cache_state=cold "
                 "validation=zero_sum_exact purpose=view1a_legacy_population_baseline\n";
    std::cout << "# storage_span_bytes is the address span required by the alias view, not a claim about unique DRAM "
                 "traffic; broadcast and overlapping views intentionally revisit addresses.\n";
    std::cout << "# The first ten cases are expected to use the legacy arbitrary-view fallback in ThorKernels308. "
                 "The final three are direct/dense controls that must not be stolen by VIEW-1.\n";
    std::cout << "family,case,dimensions,strides,axes,dtype,operation,path,strategy,physical_dense_permutation,"
                 "permutation_tiled_candidate,index_bits,rank,output_elements,reduction_elements_per_output,"
                 "workspace_bytes,logical_input_bytes,storage_span_bytes,output_bytes,median_ms,best_ms,worst_ms,"
                 "logical_GBps,validation\n";

    for (const ViewCensusCase& benchmark_case : cases) {
        for (DataType dtype : dtypes) {
            runViewCensusCase(benchmark_case, dtype, cache_flush, stream, gpu_placement);
        }
    }
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

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--arg-x4-focused|--arg-x4-awkward-focused|--arg-census|--view-census|--reduction-census|--dense-stage-cost-calibration] [--dense-run-stage-census] [--candidate=<name>]...\n"
              << "       " << executable << " --list-reduction-candidates\n"
              << "  --arg-x4-focused          Run ARGMIN only for FP8 E4M3/FP16/FP32, R=64/256/1024, and "
                 "D=128/256/512/1024/2048/4096/65536.\n"
              << "  --arg-x4-awkward-focused  Run ARGMIN exact/awkward A/B pairs for FP8 E4M3/FP16/FP32, "
                 "R=64/256/1024, and D=4096/4097/8192/8193/65536/65537.\n"
              << "  --arg-census              Run the cache-cold dense ArgMin/ArgMax performance census. Measures every "
                 "production direct ARG family plus explicit composed-dense execution for disjoint reduced runs; "
                 "composed results must exactly match production indices before timing.\n"
              << "  --view-census             Run the VIEW-1A cache-cold SUM census over remaining rank>1 arbitrary views "
                 "plus direct/dense controls. Records logical dimensions/strides, production path, storage span, "
                 "index width, workspace, and traversal throughput without changing production selection.\n"
              << "  --reduction-census        Run exact cache-cold SUM cases spanning the current production dense "
                 "value-reduction families. Production path drift is treated as an error.\n"
              << "  --dense-run-stage-census With --reduction-census, additionally time every reachable direct stage "
                 "of the dense composed cases both cache-cold and L2-hot. This is planner-model evidence only; "
                 "production selection is unchanged.\n"
              << "  --dense-stage-cost-calibration  Benchmark a bounded direct-reducer geometry grid for fitting the "
                 "stamp-time composed-dense stage cost model. Emits cold and hot timings and never changes production "
                 "selection.\n"
              << "  --candidate=<name>        With --reduction-census, also run a registered benchmark-only "
                 "candidate on every census case it supports. Repeat the flag to compare multiple candidates in one "
                 "run. Production selection is unchanged.\n"
              << "  --list-reduction-candidates  List benchmark-only candidates linked into this executable.\n";
}

BenchmarkOptions parseOptions(int argc, char** argv) {
    BenchmarkOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--arg-x4-focused") {
            options.focused_arg_x4 = true;
        } else if (argument == "--arg-x4-awkward-focused") {
            options.focused_arg_x4_awkward = true;
        } else if (argument == "--arg-census") {
            options.arg_census = true;
        } else if (argument == "--view-census") {
            options.view_census = true;
        } else if (argument == "--reduction-census") {
            options.reduction_census = true;
        } else if (argument == "--dense-run-stage-census") {
            options.dense_run_stage_census = true;
        } else if (argument == "--dense-stage-cost-calibration") {
            options.dense_stage_cost_calibration = true;
        } else if (argument == "--list-reduction-candidates") {
            options.list_reduction_candidates = true;
        } else if (argument.starts_with("--candidate=")) {
            const std::string_view name = argument.substr(std::string_view("--candidate=").size());
            if (name.empty()) {
                throw std::invalid_argument("--candidate requires a non-empty candidate name.");
            }
            options.candidate_names.emplace_back(name);
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("Unknown benchmark argument: " + std::string(argument));
        }
    }

    const int mode_count = static_cast<int>(options.focused_arg_x4) + static_cast<int>(options.focused_arg_x4_awkward)
                           + static_cast<int>(options.arg_census) + static_cast<int>(options.view_census)
                           + static_cast<int>(options.reduction_census) + static_cast<int>(options.dense_stage_cost_calibration);
    if (mode_count > 1) {
        throw std::invalid_argument("Select at most one focused benchmark mode.");
    }
    if (!options.candidate_names.empty() && !options.reduction_census) {
        throw std::invalid_argument("--candidate may only be used with --reduction-census.");
    }
    if (options.dense_run_stage_census && !options.reduction_census) {
        throw std::invalid_argument("--dense-run-stage-census may only be used with --reduction-census.");
    }
    if (options.list_reduction_candidates
        && (mode_count != 0 || options.dense_run_stage_census || !options.candidate_names.empty())) {
        throw std::invalid_argument("--list-reduction-candidates must be used by itself.");
    }
    return options;
}

int main(int argc, char** argv) {
    BenchmarkOptions options;
    try {
        options = parseOptions(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (options.list_reduction_candidates) {
        const auto& candidates = getReductionCandidates();
        if (candidates.empty()) {
            std::cout << "# no benchmark-only reduction candidates registered\n";
        } else {
            for (const std::unique_ptr<ReductionCandidate>& candidate : candidates) {
                std::cout << candidate->getName() << '\n';
            }
        }
        return EXIT_SUCCESS;
    }

    std::vector<const ReductionCandidate*> selected_candidates;
    selected_candidates.reserve(options.candidate_names.size());
    for (const std::string& candidate_name : options.candidate_names) {
        const ReductionCandidate* candidate = findReductionCandidate(candidate_name);
        if (candidate == nullptr) {
            std::cerr << "Unknown reduction benchmark candidate '" << candidate_name
                      << "'. Use --list-reduction-candidates to inspect linked candidates.\n";
            return EXIT_FAILURE;
        }
        if (std::find(selected_candidates.begin(), selected_candidates.end(), candidate) != selected_candidates.end()) {
            std::cerr << "Reduction benchmark candidate '" << candidate_name << "' was selected more than once.\n";
            return EXIT_FAILURE;
        }
        selected_candidates.push_back(candidate);
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
        if (options.reduction_census || options.arg_census || options.view_census || options.dense_stage_cost_calibration) {
            std::cout << "# A separate >=8x-L2 cache-flush buffer is touched outside every cache-cold timed sample.\n";
            if (options.reduction_census) {
                std::cout << "# Exact production-census shapes are preserved. Experimental candidates are benchmark-only "
                             "and do not participate in CubReduction production path selection.\n";
            }
        } else {
            std::cout << "# Each timed reduction reads an input >= target_input_bytes. Because the input is >= 8x L2, "
                         "successive iterations cannot benchmark an L2-resident working set.\n";
        }
        std::cout << "# timing_samples=" << TIMING_SAMPLES;
        if (options.reduction_census || options.arg_census || options.view_census || options.dense_stage_cost_calibration) {
            std::cout << " timed_iterations_per_sample=1 reported_time=median\n";
        } else {
            std::cout << " timed_iterations_per_sample=" << TIMED_ITERATIONS_PER_SAMPLE
                      << " reported_time=median\n";
        }
        std::cout << "# argmin_index/argmax_index benchmark the production index-only UINT32 path using randomized "
                     "finite input initialized outside the timed interval.\n";
        if (options.focused_arg_x4) {
            std::cout << "# mode=arg_x4_focused operations=argmin_index "
                         "dtypes=fp8_e4m3|fp16|fp32 reductions=64|256|1024 "
                         "inners=128|256|512|1024|2048|4096|65536\n";
        } else if (options.focused_arg_x4_awkward) {
            std::cout << "# mode=arg_x4_awkward_focused operations=argmin_index "
                         "dtypes=fp8_e4m3|fp16|fp32 reductions=64|256|1024 "
                         "inners=4096|4097|8192|8193|65536|65537\n";
        }
        TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, device);
        Stream stream(device);

        if (options.dense_stage_cost_calibration) {
            Tensor cache_flush(gpu_placement, TensorDescriptor(DataType::UINT8, {target_input_bytes}));
            runDenseStageCostCalibration(cache_flush, stream, gpu_placement);
            return EXIT_SUCCESS;
        }
        if (options.arg_census) {
            Tensor cache_flush(gpu_placement, TensorDescriptor(DataType::UINT8, {target_input_bytes}));
            runArgCensus(cache_flush, stream, gpu_placement);
            return EXIT_SUCCESS;
        }
        if (options.view_census) {
            Tensor cache_flush(gpu_placement, TensorDescriptor(DataType::UINT8, {target_input_bytes}));
            runViewCensus(cache_flush, stream, gpu_placement);
            return EXIT_SUCCESS;
        }
        if (!options.reduction_census) {
            std::cout << "shape,dtype,operation,path,strategy,outer,reduction,inner,input_bytes,output_bytes,median_ms,"
                         "best_ms,worst_ms,logical_GBps\n";
        }

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

        if (options.reduction_census) {
            std::cout << "# mode=reduction_census operation=sum dtypes=fp16|bf16|fp32 cache_flush_bytes="
                      << target_input_bytes << '\n';
            for (const ReductionCandidate* candidate : selected_candidates) {
                std::cout << "# candidate=" << candidate->getName()
                          << " unsupported_cases=omitted production_selector=unchanged\n";
            }
            std::cout << "# Production launch-policy fields are backend_managed when the current backend does not "
                         "publish stamped launch metadata. Candidates should report them when they own the policy.\n";
            std::cout << "executor,family,case,dimensions,axes,dtype,operation,implementation,strategy,index_bits,"
                         "output_elements,reduction_elements_per_output,vector_elements_per_load,block_threads,"
                         "first_stage_blocks,shards_per_output,scratch_bytes,input_bytes,output_bytes,median_ms,best_ms,worst_ms,"
                         "logical_GBps\n";

            // This census is intentionally small and exact. It locks down the current production selector while new
            // reducer families are developed side-by-side. A benchmark-only candidate may support any subset of these
            // cases without changing CubReductionPath or production stamping.
            const std::vector<ExactReductionCase> exact_cases = {
                {"full_dense",
                 "full_reduce_large",
                 {1024, 32768},
                 {0, 1},
                 CubReductionPath::DeviceTransformReduce},
                {"contiguous_suffix",
                 "contiguous_suffix_large",
                 {4096, 8192},
                 {1},
                 CubReductionPath::ContiguousFixedSegment},
                {"tiled_middle",
                 "tiled_middle_large",
                 {256, 512, 256},
                 {1},
                 CubReductionPath::TiledFixedSegment},
                // Direct-vs-async Tiled probes. These are exact stage-cost calibration geometries where
                // FP16/BF16 async full-row throughput collapsed while FP32 remained fast. The benchmark-only
                // tiled_direct_component_fallback candidate forces Thor's existing direct component-tiled backend.
                {"tiled_async_probe",
                 "async_x8_r127_i193",
                 {5476, 127, 193},
                 {1},
                 CubReductionPath::TiledFixedSegment},
                {"tiled_async_probe",
                 "async_x16_r127_i383",
                 {2760, 127, 383},
                 {1},
                 CubReductionPath::TiledFixedSegment},
                {"tiled_async_probe",
                 "async_group2_r127_i769",
                 {1375, 127, 769},
                 {1},
                 CubReductionPath::TiledFixedSegment},
                {"tiled_async_probe",
                 "async_group4_r127_i1537",
                 {688, 127, 1537},
                 {1},
                 CubReductionPath::TiledFixedSegment},
                {"tiled_async_probe",
                 "async_group8_r127_i2813",
                 {376, 127, 2813},
                 {1},
                 CubReductionPath::TiledFixedSegment},
                // Few outputs, enormous reduction, trailing reduced physical run: R K R.
                {"dense_runs_trailing_reduced",
                 "alexnet_conv1_bias",
                 {512, 64, 55, 55},
                 {0, 2, 3},
                 CubReductionPath::ComposedDense},
                // Alternating dense runs with a trailing retained physical run: R K R K.
                {"dense_runs_trailing_retained",
                 "rk_rk_awkward",
                 {257, 63, 31, 65},
                 {0, 2},
                 CubReductionPath::ComposedDense},
                // Fully alternating dense runs ending reduced: R K R K R.
                {"dense_runs_alternating_trailing_reduced",
                 "rkrkr_awkward",
                 {31, 17, 29, 19, 23},
                 {0, 2, 4},
                 CubReductionPath::ComposedDense},
                // Many outputs with a comparatively short reduction, ending retained: K R K R K.
                {"dense_runs_alternating_trailing_retained",
                 "krkrk_many_outputs",
                 {17, 31, 19, 29, 23},
                 {1, 3},
                 CubReductionPath::ComposedDense},
                // Ordering probe: same R K R K topology as rk_rk_awkward, but the later reduced run is
                // deliberately much larger. This separates "left-to-right" from "reduce the larger run first".
                {"dense_runs_order_probe",
                 "rkrk_later_reduction_large",
                 {31, 63, 257, 65},
                 {0, 2},
                 CubReductionPath::ComposedDense},
                // Ordering probe: K R K R K with a much larger later reduced run.
                {"dense_runs_order_probe",
                 "krkrk_later_reduction_large",
                 {17, 29, 19, 127, 23},
                 {1, 3},
                 CubReductionPath::ComposedDense},
                // Ordering probe: R K R ending reduced, but the leading run is much larger than the contiguous suffix.
                // This separates "right-to-left" from "reduce the larger run first".
                {"dense_runs_order_probe",
                 "rkr_leading_reduction_large",
                 {3025, 64, 512},
                 {0, 2},
                 CubReductionPath::ComposedDense},
                // Ordering probe: three reduced runs ending reduced, with a deliberately small suffix reduction.
                {"dense_runs_order_probe",
                 "rkrkr_small_suffix",
                 {127, 17, 97, 19, 7},
                 {0, 2, 4},
                 CubReductionPath::ComposedDense},
                // Three reduced runs ending retained. Tests whether the left-to-right policy continues to hold once
                // there are more than two composition choices.
                {"dense_runs_order_probe",
                 "rkrkrk_three_reduced_runs",
                 {61, 7, 53, 11, 31, 13},
                 {0, 2, 4},
                 CubReductionPath::ComposedDense},
                // Three reduced runs ending reduced. This is the complementary right-to-left multi-stage case.
                {"dense_runs_order_probe",
                 "krkrkr_three_reduced_runs",
                 {7, 61, 11, 53, 13, 31},
                 {1, 3, 5},
                 CubReductionPath::ComposedDense},
                // Four reduced runs exercise a longer right-to-left composition than any prior census case.
                {"dense_runs_order_probe",
                 "rkrkrkr_four_reduced_runs",
                 {29, 5, 23, 7, 19, 11, 13},
                 {0, 2, 4, 6},
                 CubReductionPath::ComposedDense},
                // Endpoint-class probe: two reduced runs, retained prefix and reduced suffix (K R K R).
                // The previous heuristic chooses right-to-left solely because the tensor ends reduced; this case checks
                // whether a retained leading run instead makes left-to-right the stable choice.
                {"dense_runs_order_probe",
                 "krkr_two_reduced_runs",
                 {31, 127, 29, 97},
                 {1, 3},
                 CubReductionPath::ComposedDense},
                // Endpoint-class probe: four reduced runs with retained prefix and reduced suffix (K R K R K R K R).
                // This extends the same K...R question to a deeper composition and rank 8.
                {"dense_runs_order_probe",
                 "krkrkrkr_four_reduced_runs",
                 {2, 29, 3, 23, 5, 19, 7, 13},
                 {1, 3, 5, 7},
                 CubReductionPath::ComposedDense},
                // Multi-axis reduced-run probe: K | RR | K | RR.  Physical run count is still two reduced runs, but
                // each reduction stage spans multiple logical dimensions.
                {"dense_runs_order_probe",
                 "krrkrr_multi_axis_runs",
                 {17, 31, 7, 19, 29, 5},
                 {1, 2, 4, 5},
                 CubReductionPath::ComposedDense},
                // Complementary multi-axis endpoint probe: RR | K | RR | K | RR.  Both physical endpoints are reduced,
                // exercising the proposed R...R right-to-left rule with multi-dimensional reduction spans.
                {"dense_runs_order_probe",
                 "rrkrrkrr_multi_axis_runs",
                 {31, 7, 19, 29, 5, 23, 3},
                 {0, 1, 3, 4, 6},
                 CubReductionPath::ComposedDense},
            };
            const std::vector<DataType> exact_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
            Tensor cache_flush(gpu_placement, TensorDescriptor(DataType::UINT8, {target_input_bytes}));
            for (const ExactReductionCase& benchmark_case : exact_cases) {
                for (DataType dtype : exact_dtypes) {
                    Tensor input(gpu_placement, TensorDescriptor(dtype, benchmark_case.dimensions));
                    if (!selected_candidates.empty()) {
                        // Candidate runs get non-trivial finite data so the out-of-band production comparison can
                        // reject missing writes, bad indexing, and incorrect partial-reduction assembly. Initialization
                        // remains completely outside every timed interval.
                        input.fillRandom(-0.25, 0.25, stream);
                    } else {
                        checkCuda(cudaMemsetAsync(input.getMemPtr<void>(), 0, input.getArraySizeInBytes(), stream.getStream()),
                                  "cudaMemsetAsync(input)");
                    }
                    stream.synchronize();

                    runProductionExactCase(
                        benchmark_case, dtype, CubReductionOp::Sum, input, cache_flush, stream);
                    for (const ReductionCandidate* candidate : selected_candidates) {
                        runCandidateExactCase(benchmark_case,
                                              dtype,
                                              CubReductionOp::Sum,
                                              *candidate,
                                              input,
                                              cache_flush,
                                              stream);
                    }
                }
            }
            if (options.dense_run_stage_census) {
                runDenseRunStageCensus(exact_cases, exact_dtypes, cache_flush, stream, gpu_placement);
            }
            return EXIT_SUCCESS;
        }

        for (const ReductionShape& shape : shapes) {
            if (options.focused_arg_x4 && !isFocusedArgX4Shape(shape.name)) {
                continue;
            }
            if (options.focused_arg_x4_awkward && !isFocusedArgX4AwkwardShape(shape.name)) {
                continue;
            }
            for (DataType dtype : dtypes) {
                if ((options.focused_arg_x4 || options.focused_arg_x4_awkward) && !isFocusedArgX4DType(dtype)) {
                    continue;
                }
                if (!options.focused_arg_x4 && !options.focused_arg_x4_awkward) {
                    for (CubReductionOp op : operations) {
                        runCase(shape, dtype, op, target_input_bytes, stream, gpu_placement);
                    }
                }
                for (CubArgReductionOp op : arg_operations) {
                    if ((options.focused_arg_x4 || options.focused_arg_x4_awkward) && op != CubArgReductionOp::ArgMin) {
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
