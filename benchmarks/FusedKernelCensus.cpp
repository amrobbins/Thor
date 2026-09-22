#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr uint64_t DEFAULT_L2_WORKING_SET_MULTIPLE = 8;
constexpr uint32_t DEFAULT_MAX_ROTATION_SLOTS = 64;
constexpr int DEFAULT_WARMUP_ROUNDS = 2;
constexpr int DEFAULT_TIMING_SAMPLES = 7;

struct BenchmarkOptions {
    int device = 0;
    uint64_t l2_working_set_multiple = DEFAULT_L2_WORKING_SET_MULTIPLE;
    uint32_t max_rotation_slots = DEFAULT_MAX_ROTATION_SLOTS;
    int warmup_rounds = DEFAULT_WARMUP_ROUNDS;
    int timing_samples = DEFAULT_TIMING_SAMPLES;
    std::optional<std::string> family_filter;
    std::optional<std::string> case_filter;
    bool list_cases = false;
};

struct InputSpec {
    std::string name;
    DataType dtype = DataType::FP32;
    std::vector<uint64_t> dimensions;
    // Number of elements that the target fused kernel can actually touch in one
    // launch.  This may be smaller than the backing allocation for strided-view
    // census cases.  std::nullopt means the whole tensor is touched.
    std::optional<uint64_t> compulsory_elements;
    bool structural_metadata = false;
    std::optional<uint32_t> scalar_u32_value;
};

struct CensusCase {
    std::string family;
    std::string name;
    std::vector<InputSpec> inputs;
    std::vector<uint64_t> output_dimensions;
    DataType output_dtype = DataType::FP32;
    // Root tensor operands logically read per output element.  This deliberately
    // preserves the historical/effective bandwidth interpretation for broadcast
    // cases.  Structural metadata inputs are excluded.
    std::vector<DataType> effective_input_dtypes;
    std::function<Expression()> build_output;
};

struct TimingSummary {
    double best_ms = 0.0;
    double median_ms = 0.0;
    double worst_ms = 0.0;
};

struct PoolSlot {
    std::unordered_map<std::string, Tensor> inputs;
    std::shared_ptr<StampedExecutionPlan> plan;
};

struct LaunchGeometry {
    uint32_t grid_x = 1;
    uint32_t grid_y = 1;
    uint32_t grid_z = 1;
    uint32_t block_x = 1;
    uint32_t block_y = 1;
    uint32_t block_z = 1;
};

[[nodiscard]] uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* where) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string(where) + " overflowed uint64_t");
    }
    return lhs * rhs;
}

[[nodiscard]] uint64_t checkedAdd(uint64_t lhs, uint64_t rhs, const char* where) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        throw std::overflow_error(std::string(where) + " overflowed uint64_t");
    }
    return lhs + rhs;
}

[[nodiscard]] uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) {
        throw std::invalid_argument("ceilDiv denominator must be non-zero");
    }
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] uint64_t numElements(const std::vector<uint64_t>& dimensions) {
    uint64_t total = 1;
    for (uint64_t dimension : dimensions) {
        total = checkedMultiply(total, dimension, "numElements");
    }
    return total;
}

[[nodiscard]] uint64_t dataTypeBytes(DataType dtype) {
    return static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
}

[[nodiscard]] std::string dataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "fp16";
        case DataType::FP32:
            return "fp32";
        case DataType::BF16:
            return "bf16";
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        case DataType::UINT32:
            return "uint32";
        case DataType::UINT64:
            return "uint64";
        default:
            return "dtype_" + std::to_string(static_cast<int>(dtype));
    }
}

[[nodiscard]] std::string dimensionsString(const std::vector<uint64_t>& dimensions) {
    std::ostringstream out;
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) out << 'x';
        out << dimensions[i];
    }
    return out.str();
}

[[nodiscard]] std::string inputShapesString(const CensusCase& c) {
    std::ostringstream out;
    bool first = true;
    for (const InputSpec& input : c.inputs) {
        if (!first) out << '|';
        first = false;
        out << input.name << ':' << dataTypeName(input.dtype) << ':' << dimensionsString(input.dimensions);
    }
    return out.str();
}

[[nodiscard]] std::string inputDTypesString(const CensusCase& c) {
    std::ostringstream out;
    bool first = true;
    for (const InputSpec& input : c.inputs) {
        if (input.structural_metadata) continue;
        if (!first) out << '|';
        first = false;
        out << dataTypeName(input.dtype);
    }
    return out.str();
}

[[nodiscard]] std::string launchKindName(CompiledEquation::LaunchKind kind) {
    switch (kind) {
        case CompiledEquation::LaunchKind::Flat:
            return "flat";
        case CompiledEquation::LaunchKind::BroadcastSingle:
            return "broadcast_single";
        case CompiledEquation::LaunchKind::BroadcastGrouped:
            return "broadcast_grouped";
        case CompiledEquation::LaunchKind::FusedTiledTranspose:
            return "fused_tiled_transpose";
    }
    return "unknown";
}

[[nodiscard]] std::string runtimeExtentSourceName(RaggedRuntimeExtentSource source) {
    switch (source) {
        case RaggedRuntimeExtentSource::DEVICE_OFFSETS:
            return "device_offsets";
        case RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT:
            return "device_active_count";
        case RaggedRuntimeExtentSource::HOST_EXTENT:
            return "host_extent";
    }
    return "unknown";
}

[[nodiscard]] std::string selectedPathName(const CompiledEquation& compiled) {
    if (compiled.launch_kind == CompiledEquation::LaunchKind::FusedTiledTranspose) {
        return "fused_tiled_transpose";
    }
    if (compiled.launch_kind == CompiledEquation::LaunchKind::BroadcastSingle ||
        compiled.launch_kind == CompiledEquation::LaunchKind::BroadcastGrouped) {
        if (compiled.uses_device_runtime_extent) {
            return compiled.elements_per_thread > 1 ? "ragged_broadcast_vector" : "ragged_broadcast_scalar";
        }
        return compiled.elements_per_thread > 1 ? "broadcast_vector" : "broadcast_scalar";
    }
    if (compiled.uses_device_runtime_extent) {
        return compiled.elements_per_thread > 1 ? "ragged_flat_vector" : "ragged_flat_scalar";
    }
    return compiled.elements_per_thread > 1 ? "flat_vector" : "flat_scalar";
}

[[nodiscard]] uint32_t packetScalars(const CompiledEquation& compiled) {
    if (compiled.launch_kind == CompiledEquation::LaunchKind::FusedTiledTranspose) {
        return std::max<uint32_t>(1, compiled.tiled_transpose_pack_scalars);
    }
    return std::max<uint32_t>(1, compiled.elements_per_thread);
}

[[nodiscard]] std::string nominalInputPacketBytes(const CensusCase& c, const CompiledEquation& compiled) {
    const uint32_t scalars = packetScalars(compiled);
    std::ostringstream out;
    bool first = true;
    for (const InputSpec& input : c.inputs) {
        if (input.structural_metadata) continue;
        if (!first) out << '|';
        first = false;
        out << checkedMultiply(scalars, dataTypeBytes(input.dtype), "nominal input packet bytes");
    }
    return out.str();
}

[[nodiscard]] uint64_t outputPacketBytes(const CensusCase& c, const CompiledEquation& compiled) {
    return checkedMultiply(packetScalars(compiled), dataTypeBytes(c.output_dtype), "output packet bytes");
}

void checkCuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

void checkCu(CUresult status, const char* where) {
    if (status == CUDA_SUCCESS) return;
    const char* error_name = nullptr;
    const char* error_string = nullptr;
    (void)cuGetErrorName(status, &error_name);
    (void)cuGetErrorString(status, &error_string);
    throw std::runtime_error(std::string(where) + ": " +
                             (error_name != nullptr ? error_name : "CUDA_ERROR") + " (" +
                             (error_string != nullptr ? error_string : "unknown driver error") + ")");
}

[[nodiscard]] int functionAttribute(CUfunction function, CUfunction_attribute attribute, const char* where) {
    int value = 0;
    checkCu(cuFuncGetAttribute(&value, attribute, function), where);
    return value;
}

[[nodiscard]] LaunchGeometry launchGeometry(const CensusCase& c, const CompiledEquation& compiled) {
    LaunchGeometry geometry;
    if (compiled.launch_kind == CompiledEquation::LaunchKind::FusedTiledTranspose) {
        constexpr uint32_t TILE_DIM = 32;
        constexpr uint32_t BLOCK_ROWS = 8;
        constexpr uint64_t MAX_CUDA_GRID_YZ = 65535ULL;
        if (c.output_dimensions.size() < 2) {
            throw std::runtime_error("Fused tiled-transpose census case requires rank >= 2 output");
        }
        uint64_t batch_count = 1;
        for (size_t i = 0; i + 2 < c.output_dimensions.size(); ++i) {
            batch_count = checkedMultiply(batch_count, c.output_dimensions[i], "transpose batch count");
        }
        const uint64_t num_rows = c.output_dimensions[c.output_dimensions.size() - 1];
        const uint64_t num_cols = c.output_dimensions[c.output_dimensions.size() - 2];
        const uint64_t tile_col_scalars = checkedMultiply(TILE_DIM, packetScalars(compiled), "transpose tile columns");
        const uint64_t row_tiles = ceilDiv(num_rows, TILE_DIM);
        const uint64_t flat_grid_y = checkedMultiply(batch_count, row_tiles, "transpose flat grid y");
        const uint64_t grid_y = std::max<uint64_t>(1, std::min<uint64_t>(flat_grid_y, MAX_CUDA_GRID_YZ));
        const uint64_t grid_z = flat_grid_y == 0 ? 1 : ceilDiv(flat_grid_y, grid_y);
        geometry.grid_x = static_cast<uint32_t>(ceilDiv(num_cols, tile_col_scalars));
        geometry.grid_y = static_cast<uint32_t>(grid_y);
        geometry.grid_z = static_cast<uint32_t>(grid_z);
        geometry.block_x = TILE_DIM;
        geometry.block_y = BLOCK_ROWS;
        return geometry;
    }

    const uint64_t max_numel = numElements(c.output_dimensions);
    const uint64_t launch_numel = ceilDiv(max_numel, std::max<uint32_t>(1, compiled.elements_per_thread));
    geometry.block_x = static_cast<uint32_t>(std::min<uint64_t>(launch_numel, 256ULL));
    uint64_t grid_x = ceilDiv(launch_numel, geometry.block_x);
    if (compiled.uses_device_runtime_extent) grid_x = std::min<uint64_t>(grid_x, 256ULL);
    geometry.grid_x = static_cast<uint32_t>(grid_x);
    return geometry;
}

__global__ void readForL2EvictionKernel(const uint4* __restrict__ words,
                                        uint64_t word_count,
                                        unsigned long long* __restrict__ sink) {
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    unsigned long long local = 0;
    for (; idx < word_count; idx += stride) {
        const uint4 v = words[idx];
        local += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w;
    }
    // Fold every lane's accumulator into lane 0 before the atomic.  This makes
    // every thread's loads observable to the optimizer while keeping the global
    // side effect to one atomic per warp.  The eviction pass is always outside
    // the timed target interval.
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffffU, local, offset);
    }
    if ((threadIdx.x & 31U) == 0U) atomicAdd(sink, local);
}

class L2Evictor {
   public:
    L2Evictor() = default;

    L2Evictor(uint64_t bytes, int device) : bytes_(bytes) {
        if (bytes_ == 0) return;
        ScopedGpu scoped_gpu(device);
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, device);
        const uint64_t rounded_bytes = ((bytes_ + 15ULL) / 16ULL) * 16ULL;
        bytes_ = rounded_bytes;
        storage_ = Tensor(placement, TensorDescriptor(DataType::UINT32, {bytes_ / sizeof(uint32_t)}));
        sink_ = Tensor(placement, TensorDescriptor(DataType::UINT64, {1}));
    }

    void prime(Stream& stream) {
        if (bytes_ == 0) return;
        checkCuda(cudaMemsetAsync(storage_.getMemPtr<void>(), 0x5a, storage_.getArraySizeInBytes(), stream.getStream()),
                  "cudaMemsetAsync(L2 eviction storage)");
        checkCuda(cudaMemsetAsync(sink_.getMemPtr<void>(), 0, sink_.getArraySizeInBytes(), stream.getStream()),
                  "cudaMemsetAsync(L2 eviction sink)");
        stream.synchronize();
    }

    void evict(Stream& stream) {
        if (bytes_ == 0) return;
        constexpr uint32_t threads = 256;
        const uint64_t word_count = bytes_ / sizeof(uint4);
        int multiprocessor_count = 0;
        checkCuda(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, stream.getGpuNum()),
                  "cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)");
        const uint64_t requested_blocks = ceilDiv(word_count, threads);
        const uint64_t useful_blocks = static_cast<uint64_t>(std::max(1, multiprocessor_count)) * 8ULL;
        const uint32_t blocks = static_cast<uint32_t>(std::max<uint64_t>(1, std::min(requested_blocks, useful_blocks)));
        readForL2EvictionKernel<<<blocks, threads, 0, stream.getStream()>>>(
            reinterpret_cast<const uint4*>(storage_.getMemPtr<void>()),
            word_count,
            reinterpret_cast<unsigned long long*>(sink_.getMemPtr<void>()));
        checkCuda(cudaGetLastError(), "readForL2EvictionKernel launch");
    }


   private:
    uint64_t bytes_ = 0;
    Tensor storage_;
    Tensor sink_;
};

[[nodiscard]] Expression typedInput(const std::string& name, DataType storage_dtype) {
    return Expression::input(name, DataType::FP32, storage_dtype);
}

[[nodiscard]] Expression arithmeticBinary(const Expression& x, const Expression& y, DataType output_dtype) {
    return ((x * Expression(1.25)) + (y * Expression(-0.5))).withOutputDType(output_dtype);
}

[[nodiscard]] Expression arithmeticDeep(const Expression& x, const Expression& y, DataType output_dtype) {
    return (((x * Expression(1.25)) + (y * Expression(-0.5)) + ((x - y) * Expression(0.125)) - Expression(0.25))
                .swish() *
            Expression(0.75))
        .withOutputDType(output_dtype);
}

[[nodiscard]] CensusCase makeDenseUnaryCase(std::string name,
                                            DataType dtype,
                                            const std::vector<uint64_t>& shape) {
    CensusCase c;
    c.family = "dense";
    c.name = std::move(name);
    c.inputs = {{"x", dtype, shape, std::nullopt, false, std::nullopt}};
    c.output_dimensions = shape;
    c.output_dtype = dtype;
    c.effective_input_dtypes = {dtype};
    c.build_output = [dtype] {
        const Expression x = typedInput("x", dtype);
        return (x * Expression(1.125)).relu().withOutputDType(dtype);
    };
    return c;
}

[[nodiscard]] CensusCase makeDenseBinaryCase(std::string name,
                                             DataType lhs_dtype,
                                             DataType rhs_dtype,
                                             DataType output_dtype,
                                             const std::vector<uint64_t>& shape,
                                             bool deep) {
    CensusCase c;
    c.family = lhs_dtype == rhs_dtype && lhs_dtype == output_dtype ? "dense" : "mixed";
    c.name = std::move(name);
    c.inputs = {
        {"x", lhs_dtype, shape, std::nullopt, false, std::nullopt},
        {"y", rhs_dtype, shape, std::nullopt, false, std::nullopt},
    };
    c.output_dimensions = shape;
    c.output_dtype = output_dtype;
    c.effective_input_dtypes = {lhs_dtype, rhs_dtype};
    c.build_output = [lhs_dtype, rhs_dtype, output_dtype, deep] {
        const Expression x = typedInput("x", lhs_dtype);
        const Expression y = typedInput("y", rhs_dtype);
        return deep ? arithmeticDeep(x, y, output_dtype) : arithmeticBinary(x, y, output_dtype);
    };
    return c;
}

[[nodiscard]] CensusCase makeBroadcastCase(std::string name,
                                           DataType lhs_dtype,
                                           DataType rhs_dtype,
                                           DataType output_dtype,
                                           std::vector<uint64_t> lhs_shape,
                                           std::vector<uint64_t> rhs_shape,
                                           std::vector<uint64_t> output_shape,
                                           bool ragged,
                                           uint64_t ragged_batch_size = 0,
                                           uint64_t ragged_max_active_values = 0,
                                           uint64_t ragged_elements_per_value = 0) {
    CensusCase c;
    c.family = ragged ? "ragged_broadcast" :
               (lhs_dtype == rhs_dtype && lhs_dtype == output_dtype ? "broadcast" : "mixed_broadcast");
    c.name = std::move(name);
    c.inputs = {
        {"x", lhs_dtype, std::move(lhs_shape), std::nullopt, false, std::nullopt},
        {"y", rhs_dtype, std::move(rhs_shape), std::nullopt, false, std::nullopt},
    };
    if (ragged) {
        c.inputs.push_back(
            {"active_count", DataType::UINT32, {1}, 1, true, static_cast<uint32_t>(ragged_max_active_values)});
    }
    c.output_dimensions = std::move(output_shape);
    c.output_dtype = output_dtype;
    c.effective_input_dtypes = {lhs_dtype, rhs_dtype};
    c.build_output = [lhs_dtype,
                      rhs_dtype,
                      output_dtype,
                      ragged,
                      ragged_batch_size,
                      ragged_max_active_values,
                      ragged_elements_per_value] {
        const Expression x = typedInput("x", lhs_dtype);
        const Expression y = typedInput("y", rhs_dtype);
        Expression out = arithmeticBinary(x, y, output_dtype);
        if (ragged) {
            const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
            out = out.withRaggedRuntimeExtent(active_count,
                                              ragged_batch_size,
                                              ragged_max_active_values,
                                              ragged_elements_per_value,
                                              RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        }
        return out;
    };
    return c;
}

[[nodiscard]] CensusCase makeRaggedValuewiseCase(std::string name,
                                                 DataType dtype,
                                                 uint64_t active_values,
                                                 uint64_t width,
                                                 bool deep) {
    const std::vector<uint64_t> shape{active_values, width};
    CensusCase c;
    c.family = "ragged";
    c.name = std::move(name);
    c.inputs = {
        {"x", dtype, shape, std::nullopt, false, std::nullopt},
        {"y", dtype, shape, std::nullopt, false, std::nullopt},
        {"active_count", DataType::UINT32, {1}, 1, true, static_cast<uint32_t>(active_values)},
    };
    c.output_dimensions = shape;
    c.output_dtype = dtype;
    c.effective_input_dtypes = {dtype, dtype};
    c.build_output = [dtype, active_values, width, deep] {
        const Expression x = typedInput("x", dtype);
        const Expression y = typedInput("y", dtype);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        Expression out = deep ? arithmeticDeep(x, y, dtype) : arithmeticBinary(x, y, dtype);
        return out.withRaggedRuntimeExtent(active_count,
                                           128,
                                           active_values,
                                           width,
                                           RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
    };
    return c;
}

[[nodiscard]] CensusCase makeRaggedMixedValuewiseCase(std::string name,
                                                      DataType lhs_dtype,
                                                      DataType rhs_dtype,
                                                      DataType output_dtype,
                                                      uint64_t active_values,
                                                      uint64_t width) {
    const std::vector<uint64_t> shape{active_values, width};
    CensusCase c;
    c.family = "ragged_mixed";
    c.name = std::move(name);
    c.inputs = {
        {"x", lhs_dtype, shape, std::nullopt, false, std::nullopt},
        {"y", rhs_dtype, shape, std::nullopt, false, std::nullopt},
        {"active_count", DataType::UINT32, {1}, 1, true, static_cast<uint32_t>(active_values)},
    };
    c.output_dimensions = shape;
    c.output_dtype = output_dtype;
    c.effective_input_dtypes = {lhs_dtype, rhs_dtype};
    c.build_output = [lhs_dtype, rhs_dtype, output_dtype, active_values, width] {
        const Expression x = typedInput("x", lhs_dtype);
        const Expression y = typedInput("y", rhs_dtype);
        const Expression active_count = Expression::input("active_count", DataType::UINT32, DataType::UINT32);
        return arithmeticBinary(x, y, output_dtype)
            .withRaggedRuntimeExtent(active_count,
                                     128,
                                     active_values,
                                     width,
                                     RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
    };
    return c;
}

[[nodiscard]] CensusCase makeIndexedInnerSpanCase(std::string name,
                                                  DataType dtype,
                                                  uint64_t rows,
                                                  uint64_t width,
                                                  uint64_t element_offset) {
    const std::vector<uint64_t> source_shape{rows, width * 2};
    const std::vector<uint64_t> output_shape{rows, width};
    CensusCase c;
    c.family = "indexed";
    c.name = std::move(name);
    c.inputs = {
        {"x_storage", dtype, source_shape, checkedMultiply(rows, width, "indexed source touched elements"), false, std::nullopt},
        {"y", dtype, output_shape, std::nullopt, false, std::nullopt},
    };
    c.output_dimensions = output_shape;
    c.output_dtype = dtype;
    c.effective_input_dtypes = {dtype, dtype};
    c.build_output = [dtype, rows, width, element_offset] {
        const Expression storage = typedInput("x_storage", dtype);
        const Expression y = typedInput("y", dtype);
        const Expression x = storage.stridedView({rows, width}, {width * 2, 1}, element_offset);
        return arithmeticBinary(x, y, dtype);
    };
    return c;
}

[[nodiscard]] CensusCase makeTransposeCase(std::string name,
                                           DataType input_dtype,
                                           DataType output_dtype,
                                           const std::vector<uint64_t>& shape) {
    CensusCase c;
    c.family = "transpose";
    c.name = std::move(name);
    c.inputs = {
        {"x", input_dtype, shape, std::nullopt, false, std::nullopt},
        {"y", input_dtype, shape, std::nullopt, false, std::nullopt},
    };
    c.output_dimensions = {shape[1], shape[0]};
    c.output_dtype = output_dtype;
    c.effective_input_dtypes = {input_dtype, input_dtype};
    c.build_output = [input_dtype, output_dtype] {
        const Expression x = typedInput("x", input_dtype);
        const Expression y = typedInput("y", input_dtype);
        return arithmeticBinary(x, y, output_dtype).transpose();
    };
    return c;
}

[[nodiscard]] std::vector<CensusCase> buildCases() {
    constexpr uint64_t DENSE_ROWS = 8192;
    constexpr uint64_t DENSE_COLS = 4096;
    constexpr uint64_t BCAST_ROWS = 4096;
    constexpr uint64_t BCAST_COLS = 4096;
    constexpr uint64_t PRODUCT_T = 128ULL * 819ULL;

    const std::vector<uint64_t> dense_shape{DENSE_ROWS, DENSE_COLS};
    const std::vector<uint64_t> awkward_shape{8193, 4097};
    const std::vector<uint64_t> bcast_output{BCAST_ROWS, BCAST_COLS};

    std::vector<CensusCase> cases;

    for (DataType dtype : {DataType::BF16, DataType::FP16, DataType::FP32}) {
        const std::string d = dataTypeName(dtype);
        cases.push_back(makeDenseUnaryCase("dense_unary_" + d, dtype, dense_shape));
        cases.push_back(makeDenseBinaryCase("dense_binary_" + d, dtype, dtype, dtype, dense_shape, false));
        cases.push_back(makeDenseBinaryCase("dense_deep_" + d, dtype, dtype, dtype, dense_shape, true));
    }
    cases.push_back(makeDenseBinaryCase("dense_tail_bf16", DataType::BF16, DataType::BF16, DataType::BF16, awkward_shape, false));
    cases.push_back(makeDenseBinaryCase("dense_tail_fp32", DataType::FP32, DataType::FP32, DataType::FP32, awkward_shape, false));

    cases.push_back(makeDenseBinaryCase("mixed_bf16_bf16_to_fp32", DataType::BF16, DataType::BF16, DataType::FP32, dense_shape, false));
    cases.push_back(makeDenseBinaryCase("mixed_fp32_fp32_to_bf16", DataType::FP32, DataType::FP32, DataType::BF16, dense_shape, false));
    cases.push_back(makeDenseBinaryCase("mixed_bf16_fp32_to_fp32", DataType::BF16, DataType::FP32, DataType::FP32, dense_shape, false));
    cases.push_back(makeDenseBinaryCase("mixed_fp32_bf16_to_bf16", DataType::FP32, DataType::BF16, DataType::BF16, dense_shape, false));

    for (DataType dtype : {DataType::BF16, DataType::FP16, DataType::FP32}) {
        const std::string d = dataTypeName(dtype);
        cases.push_back(makeBroadcastCase("broadcast_row_" + d,
                                          dtype,
                                          dtype,
                                          dtype,
                                          bcast_output,
                                          {1, BCAST_COLS},
                                          bcast_output,
                                          false));
        cases.push_back(makeBroadcastCase("broadcast_column_" + d,
                                          dtype,
                                          dtype,
                                          dtype,
                                          bcast_output,
                                          {BCAST_ROWS, 1},
                                          bcast_output,
                                          false));
        cases.push_back(makeBroadcastCase("broadcast_outer_" + d,
                                          dtype,
                                          dtype,
                                          dtype,
                                          {BCAST_ROWS, 1},
                                          {1, BCAST_COLS},
                                          bcast_output,
                                          false));
        cases.push_back(makeBroadcastCase("broadcast_scalar_tensor_" + d,
                                          dtype,
                                          dtype,
                                          dtype,
                                          bcast_output,
                                          {1, 1},
                                          bcast_output,
                                          false));
    }
    cases.push_back(makeBroadcastCase("mixed_broadcast_row_bf16_to_fp32",
                                      DataType::BF16,
                                      DataType::BF16,
                                      DataType::FP32,
                                      bcast_output,
                                      {1, BCAST_COLS},
                                      bcast_output,
                                      false));
    cases.push_back(makeBroadcastCase("mixed_broadcast_row_bf16_fp32_to_fp32",
                                      DataType::BF16,
                                      DataType::FP32,
                                      DataType::FP32,
                                      bcast_output,
                                      {1, BCAST_COLS},
                                      bcast_output,
                                      false));

    // Product-transformer exact T scale and the channel widths called out in the
    // handoff.  These are packed runtime-extent valuewise kernels, not padded
    // ragged storage.
    for (uint64_t width : {80ULL, 128ULL, 256ULL, 384ULL, 512ULL}) {
        cases.push_back(makeRaggedValuewiseCase(
            "ragged_product_bf16_t104832_w" + std::to_string(width), DataType::BF16, PRODUCT_T, width, false));
    }
    cases.push_back(makeRaggedValuewiseCase("ragged_product_bf16_deep_t104832_w128", DataType::BF16, PRODUCT_T, 128, true));
    cases.push_back(makeRaggedValuewiseCase("ragged_small_bf16_t512_w128", DataType::BF16, 512, 128, false));
    cases.push_back(makeRaggedValuewiseCase("ragged_medium_bf16_t8192_w128", DataType::BF16, 8192, 128, false));
    cases.push_back(makeRaggedValuewiseCase("ragged_awkward_bf16_t16385_w127", DataType::BF16, 16385, 127, false));
    cases.push_back(makeRaggedValuewiseCase("ragged_product_fp16_t104832_w128", DataType::FP16, PRODUCT_T, 128, false));
    cases.push_back(makeRaggedValuewiseCase("ragged_product_fp32_t104832_w128", DataType::FP32, PRODUCT_T, 128, false));
    cases.push_back(makeRaggedMixedValuewiseCase("ragged_mixed_bf16_bf16_to_fp32_t104832_w128",
                                                  DataType::BF16,
                                                  DataType::BF16,
                                                  DataType::FP32,
                                                  PRODUCT_T,
                                                  128));
    cases.push_back(makeRaggedMixedValuewiseCase("ragged_mixed_bf16_fp32_to_fp32_t104832_w128",
                                                  DataType::BF16,
                                                  DataType::FP32,
                                                  DataType::FP32,
                                                  PRODUCT_T,
                                                  128));

    // Ragged channel/scalar broadcasts.  The first operand is the packed value;
    // the second is broadcast over the trailing width.
    for (uint64_t width : {80ULL, 128ULL, 256ULL, 384ULL, 512ULL}) {
        cases.push_back(makeBroadcastCase("ragged_broadcast_product_bf16_t104832_w" + std::to_string(width),
                                          DataType::BF16,
                                          DataType::BF16,
                                          DataType::BF16,
                                          {PRODUCT_T, width},
                                          {1, width},
                                          {PRODUCT_T, width},
                                          true,
                                          128,
                                          PRODUCT_T,
                                          width));
    }
    cases.push_back(makeBroadcastCase("ragged_broadcast_scalar_bf16_t104832_w128",
                                      DataType::BF16,
                                      DataType::BF16,
                                      DataType::BF16,
                                      {PRODUCT_T, 128},
                                      {1, 1},
                                      {PRODUCT_T, 128},
                                      true,
                                      128,
                                      PRODUCT_T,
                                      128));
    cases.push_back(makeBroadcastCase("ragged_mixed_broadcast_bf16_fp32_to_fp32_t104832_w128",
                                      DataType::BF16,
                                      DataType::FP32,
                                      DataType::FP32,
                                      {PRODUCT_T, 128},
                                      {1, 128},
                                      {PRODUCT_T, 128},
                                      true,
                                      128,
                                      PRODUCT_T,
                                      128));

    cases.push_back(makeIndexedInnerSpanCase("indexed_inner_contiguous_bf16_aligned", DataType::BF16, PRODUCT_T, 128, 0));
    cases.push_back(makeIndexedInnerSpanCase("indexed_inner_contiguous_bf16_misaligned", DataType::BF16, PRODUCT_T, 128, 1));
    cases.push_back(makeIndexedInnerSpanCase("indexed_inner_contiguous_fp32_aligned", DataType::FP32, PRODUCT_T, 128, 0));

    cases.push_back(makeTransposeCase("transpose_dense_bf16", DataType::BF16, DataType::BF16, dense_shape));
    cases.push_back(makeTransposeCase("transpose_dense_fp32", DataType::FP32, DataType::FP32, dense_shape));
    cases.push_back(makeTransposeCase("transpose_mixed_bf16_to_fp32", DataType::BF16, DataType::FP32, dense_shape));

    return cases;
}

[[nodiscard]] uint64_t compulsoryBytesPerLaunch(const CensusCase& c) {
    uint64_t bytes = checkedMultiply(numElements(c.output_dimensions), dataTypeBytes(c.output_dtype), "output compulsory bytes");
    for (const InputSpec& input : c.inputs) {
        const uint64_t elements = input.compulsory_elements.value_or(numElements(input.dimensions));
        bytes = checkedAdd(bytes,
                           checkedMultiply(elements, dataTypeBytes(input.dtype), "input compulsory bytes"),
                           "compulsory bytes");
    }
    return bytes;
}

[[nodiscard]] uint64_t allocationBytesPerSlot(const CensusCase& c) {
    uint64_t bytes = checkedMultiply(numElements(c.output_dimensions), dataTypeBytes(c.output_dtype), "output allocation bytes");
    for (const InputSpec& input : c.inputs) {
        bytes = checkedAdd(bytes,
                           checkedMultiply(numElements(input.dimensions), dataTypeBytes(input.dtype), "input allocation bytes"),
                           "allocation bytes");
    }
    return bytes;
}

[[nodiscard]] uint64_t effectiveBytesPerLaunch(const CensusCase& c) {
    const uint64_t output_elements = numElements(c.output_dimensions);
    uint64_t bytes_per_output = dataTypeBytes(c.output_dtype);
    for (DataType dtype : c.effective_input_dtypes) {
        bytes_per_output = checkedAdd(bytes_per_output, dataTypeBytes(dtype), "effective bytes per output");
    }
    return checkedMultiply(output_elements, bytes_per_output, "effective bytes per launch");
}

void initializeInput(const InputSpec& spec, Tensor& tensor, Stream& stream) {
    if (spec.scalar_u32_value.has_value()) {
        if (spec.dtype != DataType::UINT32 || numElements(spec.dimensions) != 1) {
            throw std::runtime_error("scalar_u32_value requires a one-element UINT32 tensor");
        }
        const uint32_t value = spec.scalar_u32_value.value();
        // The source is a stack scalar, so use the synchronous copy here rather
        // than depending on pageable-host cudaMemcpyAsync staging lifetime.
        checkCuda(cudaMemcpy(tensor.getMemPtr<void>(), &value, sizeof(value), cudaMemcpyHostToDevice),
                  "cudaMemcpy(census scalar u32)");
        return;
    }
    checkCuda(cudaMemsetAsync(tensor.getMemPtr<void>(), 0, tensor.getArraySizeInBytes(), stream.getStream()),
              "cudaMemsetAsync(census input)");
}

[[nodiscard]] std::unordered_map<std::string, Tensor> allocateInputs(const CensusCase& c,
                                                                     const TensorPlacement& placement,
                                                                     Stream& stream) {
    std::unordered_map<std::string, Tensor> inputs;
    inputs.reserve(c.inputs.size());
    for (const InputSpec& spec : c.inputs) {
        Tensor tensor(placement, TensorDescriptor(spec.dtype, spec.dimensions));
        initializeInput(spec, tensor, stream);
        inputs.emplace(spec.name, std::move(tensor));
    }
    return inputs;
}

[[nodiscard]] const CompiledEquation& requireSingleFusedKernel(const std::shared_ptr<CompiledOutputs>& compiled,
                                                               const CensusCase& c) {
    if (!compiled) {
        throw std::runtime_error(c.name + " compiled to a null output plan");
    }
    const CompiledEquation* fused = nullptr;
    for (const CompiledExecutionStage& stage : compiled->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
            throw std::runtime_error(c.name + " expected an isolated FusedKernel stage but compiled stage " +
                                     CompiledExecutionStage::kindToString(stage.kind));
        }
        if (fused != nullptr) {
            throw std::runtime_error(c.name + " expected one FusedKernel stage but compiled more than one");
        }
        if (!stage.flat) {
            throw std::runtime_error(c.name + " fused stage is missing its CompiledEquation payload");
        }
        fused = stage.flat.get();
    }
    if (fused == nullptr) {
        throw std::runtime_error(c.name + " did not compile a FusedKernel stage");
    }
    return *fused;
}

[[nodiscard]] TimingSummary timePool(const std::vector<PoolSlot>& pool,
                                     Stream& stream,
                                     L2Evictor& evictor,
                                     bool needs_explicit_eviction,
                                     int warmup_rounds,
                                     int timing_samples) {
    if (pool.empty()) {
        throw std::invalid_argument("Fused census timing pool must not be empty");
    }

    for (int round = 0; round < warmup_rounds; ++round) {
        for (const PoolSlot& slot : pool) {
            slot.plan->runOn(stream);
        }
    }
    stream.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    std::vector<double> sample_ms;
    sample_ms.reserve(static_cast<size_t>(timing_samples));
    size_t slot_index = 0;
    for (int sample = 0; sample < timing_samples; ++sample) {
        if (needs_explicit_eviction) {
            evictor.evict(stream);
            stream.synchronize();
        }

        // Time exactly one production fused-kernel launch. Rotating the slot
        // between samples keeps the reuse distance cache-controlled without
        // folding host submission gaps from a long launch train into one CUDA
        // event interval.
        checkCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(start)");
        pool[slot_index].plan->runOn(stream);
        checkCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        sample_ms.push_back(static_cast<double>(elapsed_ms));
        slot_index = (slot_index + 1) % pool.size();
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::sort(sample_ms.begin(), sample_ms.end());
    return TimingSummary{sample_ms.front(), sample_ms[sample_ms.size() / 2], sample_ms.back()};
}

[[nodiscard]] BenchmarkOptions parseOptions(int argc, char** argv) {
    BenchmarkOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto valueAfter = [&](std::string_view prefix) -> std::optional<std::string> {
            if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
            return std::nullopt;
        };

        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Thor fused-kernel census\n\n"
                << "Options:\n"
                << "  --device=N                 CUDA device (default 0)\n"
                << "  --family=NAME              run one family (dense, mixed, broadcast, mixed_broadcast, ragged, ragged_mixed, ragged_broadcast, indexed, transpose)\n"
                << "  --case=SUBSTRING           run cases whose names contain SUBSTRING\n"
                << "  --l2-multiple=N            rotating touched working set target / L2 (default 8)\n"
                << "  --max-rotation-slots=N     cap stamped allocation rotation (default 64); smaller cases use untimed L2 eviction\n"
                << "  --warmup-rounds=N          full untimed pool rounds (default 2)\n"
                << "  --samples=N                timed pool rounds (default 7)\n"
                << "  --list-cases                list census cases and exit\n\n"
                << "The timed interval contains only production fused-kernel launches.  The benchmark rotates\n"
                << "independent input/output allocations until their touched working set reaches the requested\n"
                << "multiple of L2.  If the slot cap is reached first, an untimed >L2 read pass is performed\n"
                << "before each timed sample.\n";
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--list-cases") {
            options.list_cases = true;
        } else if (const auto value = valueAfter("--device=")) {
            options.device = std::stoi(*value);
        } else if (const auto value = valueAfter("--family=")) {
            options.family_filter = *value;
        } else if (const auto value = valueAfter("--case=")) {
            options.case_filter = *value;
        } else if (const auto value = valueAfter("--l2-multiple=")) {
            options.l2_working_set_multiple = std::stoull(*value);
        } else if (const auto value = valueAfter("--max-rotation-slots=")) {
            options.max_rotation_slots = static_cast<uint32_t>(std::stoul(*value));
        } else if (const auto value = valueAfter("--warmup-rounds=")) {
            options.warmup_rounds = std::stoi(*value);
        } else if (const auto value = valueAfter("--samples=")) {
            options.timing_samples = std::stoi(*value);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.l2_working_set_multiple < 2) {
        throw std::invalid_argument("--l2-multiple must be >= 2");
    }
    if (options.max_rotation_slots == 0) {
        throw std::invalid_argument("--max-rotation-slots must be >= 1");
    }
    if (options.warmup_rounds < 0 || options.timing_samples <= 0) {
        throw std::invalid_argument("warmup rounds must be >= 0 and timing samples must be > 0");
    }
    return options;
}

[[nodiscard]] bool selected(const CensusCase& c, const BenchmarkOptions& options) {
    if (options.family_filter.has_value() && c.family != options.family_filter.value()) return false;
    if (options.case_filter.has_value() && c.name.find(options.case_filter.value()) == std::string::npos) return false;
    return true;
}

void runCase(const CensusCase& c,
             const BenchmarkOptions& options,
             uint64_t l2_bytes,
             Stream& stream,
             const TensorPlacement& placement) {
    const uint64_t compulsory_bytes = compulsoryBytesPerLaunch(c);
    const uint64_t allocation_bytes = allocationBytesPerSlot(c);
    const uint64_t effective_bytes = effectiveBytesPerLaunch(c);
    const uint64_t rotation_target_bytes = checkedMultiply(l2_bytes, options.l2_working_set_multiple, "L2 rotation target");
    // Reuse distance excludes the slot that is about to run: before slot N is
    // reused, only the other slots have intervened. Allocate one extra slot so
    // that the intervening touched bytes, not merely total pool size, reach the
    // requested multiple of L2.
    const uint64_t required_slots = checkedAdd(
        1, std::max<uint64_t>(1, ceilDiv(rotation_target_bytes, compulsory_bytes)), "census required rotation slots");
    const uint32_t pool_slots = static_cast<uint32_t>(
        std::min<uint64_t>(required_slots, static_cast<uint64_t>(options.max_rotation_slots)));
    const uint64_t pool_allocation_bytes = checkedMultiply(allocation_bytes, pool_slots, "census pool allocation bytes");
    const uint64_t pool_touched_bytes = checkedMultiply(compulsory_bytes, pool_slots, "census pool touched bytes");
    const uint64_t reuse_distance_bytes = checkedMultiply(
        compulsory_bytes, pool_slots > 0 ? static_cast<uint64_t>(pool_slots - 1) : 0ULL, "census reuse distance bytes");
    const bool needs_explicit_eviction = reuse_distance_bytes < rotation_target_bytes;

    uint64_t evict_bytes = 0;
    if (needs_explicit_eviction) {
        // The read pass itself should cover the full requested working-set target,
        // not merely one L2. Keep it separate from target allocations and outside timing.
        evict_bytes = rotation_target_bytes;
    }
    const uint64_t requested_bytes = checkedAdd(pool_allocation_bytes, evict_bytes, "census requested bytes");
    size_t current_free_bytes = 0;
    size_t current_total_bytes = 0;
    checkCuda(cudaMemGetInfo(&current_free_bytes, &current_total_bytes), "cudaMemGetInfo(case)");
    if (requested_bytes > static_cast<uint64_t>(current_free_bytes) * 3ULL / 4ULL) {
        std::ostringstream message;
        message << c.name << " requires approximately " << requested_bytes / MIB
                << " MiB for its cache-controlled pool/evictor, more than 75% of currently free GPU memory ("
                << current_free_bytes / MIB << " MiB). Reduce --l2-multiple or --max-rotation-slots.";
        throw std::runtime_error(message.str());
    }

    const Expression output = c.build_output();
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"out", output}}).physicalOutputs(), options.device);

    std::vector<PoolSlot> pool;
    pool.reserve(pool_slots);
    for (uint32_t slot_index = 0; slot_index < pool_slots; ++slot_index) {
        std::unordered_map<std::string, Tensor> inputs = allocateInputs(c, placement, stream);
        StampedExecutionPlan stamped = equation.stamp(inputs, stream);
        pool.push_back(PoolSlot{std::move(inputs), std::make_shared<StampedExecutionPlan>(std::move(stamped))});
    }
    stream.synchronize();

    const std::shared_ptr<CompiledOutputs> compiled_outputs = equation.compileForInputs(pool.front().inputs);
    const CompiledEquation& compiled = requireSingleFusedKernel(compiled_outputs, c);
    const std::vector<std::string> stage_names = pool.front().plan->stageKindNames();
    if (stage_names.size() != 1 || stage_names.front() != "FusedKernel") {
        throw std::runtime_error(c.name + " stamped plan is not exactly one FusedKernel stage");
    }

    const uint64_t model_logical_bytes = pool.front().plan->logicalByteCount();
    L2Evictor evictor(evict_bytes, options.device);
    evictor.prime(stream);

    const TimingSummary timing = timePool(
        pool, stream, evictor, needs_explicit_eviction, options.warmup_rounds, options.timing_samples);

    const double effective_gb_s = static_cast<double>(effective_bytes) / (timing.median_ms * 1.0e6);
    const double compulsory_gb_s = static_cast<double>(compulsory_bytes) / (timing.median_ms * 1.0e6);
    const double pool_over_l2 = static_cast<double>(pool_touched_bytes) / static_cast<double>(l2_bytes);
    const double reuse_distance_over_l2 = static_cast<double>(reuse_distance_bytes) / static_cast<double>(l2_bytes);
    const LaunchGeometry geometry = launchGeometry(c, compiled);
    const int registers_per_thread =
        functionAttribute(compiled.kernel, CU_FUNC_ATTRIBUTE_NUM_REGS, "cuFuncGetAttribute(NUM_REGS)");
    const int local_bytes_per_thread =
        functionAttribute(compiled.kernel, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, "cuFuncGetAttribute(LOCAL_SIZE_BYTES)");
    const int static_shared_bytes =
        functionAttribute(compiled.kernel, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, "cuFuncGetAttribute(SHARED_SIZE_BYTES)");

    std::cout << c.family << ',' << c.name << ',' << inputDTypesString(c) << ',' << dataTypeName(c.output_dtype) << ','
              << inputShapesString(c) << ',' << dimensionsString(c.output_dimensions) << ',' << compiled.kernel_name << ','
              << launchKindName(compiled.launch_kind) << ',' << selectedPathName(compiled) << ','
              << compiled.elements_per_thread << ',' << packetScalars(compiled) << ','
              << nominalInputPacketBytes(c, compiled) << ',' << outputPacketBytes(c, compiled) << ','
              << geometry.grid_x << ',' << geometry.grid_y << ',' << geometry.grid_z << ','
              << geometry.block_x << ',' << geometry.block_y << ',' << geometry.block_z << ','
              << registers_per_thread << ',' << local_bytes_per_thread << ',' << static_shared_bytes << ','
              << (compiled.uses_device_runtime_extent ? 1 : 0) << ','
              << (compiled.uses_device_runtime_extent ? runtimeExtentSourceName(compiled.device_runtime_extent_source) : "none") << ','
              << pool_slots << ',' << pool_touched_bytes << ',' << std::fixed << std::setprecision(2) << pool_over_l2 << ','
              << reuse_distance_bytes << ',' << reuse_distance_over_l2 << ','
              << (needs_explicit_eviction ? "rotation_plus_eviction" : "rotation") << ',' << effective_bytes << ','
              << compulsory_bytes << ',' << model_logical_bytes << ',' << std::setprecision(4) << timing.median_ms << ','
              << timing.best_ms << ',' << timing.worst_ms << ',' << std::setprecision(2) << effective_gb_s << ','
              << compulsory_gb_s << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkOptions options = parseOptions(argc, argv);
        const std::vector<CensusCase> cases = buildCases();

        if (options.list_cases) {
            for (const CensusCase& c : cases) {
                std::cout << c.family << ',' << c.name << '\n';
            }
            return EXIT_SUCCESS;
        }

        int device_count = 0;
        checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("FusedKernelCensus requires a CUDA GPU");
        }
        if (options.device < 0 || options.device >= device_count) {
            throw std::invalid_argument("Requested CUDA device is out of range");
        }

        ScopedGpu scoped_gpu(options.device);
        int l2_cache_bytes = 0;
        checkCuda(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, options.device),
                  "cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)");
        if (l2_cache_bytes <= 0) {
            throw std::runtime_error("CUDA reported a non-positive L2 size; cache-controlled census cannot run");
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, options.device), "cudaGetDeviceProperties");

        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, options.device);
        Stream stream(options.device);

        std::cout << "# Thor fused-kernel census\n"
                  << "# device=" << properties.name << " device_index=" << options.device
                  << " l2_bytes=" << l2_cache_bytes << " free_bytes=" << free_bytes << " total_bytes=" << total_bytes << '\n'
                  << "# l2_working_set_multiple=" << options.l2_working_set_multiple
                  << " max_rotation_slots=" << options.max_rotation_slots
                  << " warmup_rounds=" << options.warmup_rounds << " timing_samples=" << options.timing_samples << '\n'
                  << "# effective_bytes charges one root payload read per output element plus the output write; for broadcasts this is intentionally a logical/effective metric.\n"
                  << "# compulsory_bytes counts unique target tensor bytes touched once per launch, including structural metadata actually read and the output write.\n"
                  << "# input_packet_bytes are nominal contiguous value spans implied by packet_scalars; broadcast/indexed operands can legitimately load/reuse less or use indexed scalar traffic.\n"
                  << "# cache_control=rotation means the intervening reuse distance itself is >= requested L2 multiple; rotation_plus_eviction adds an untimed device-read eviction pass before each timed sample.\n"
                  << "# each timing sample contains exactly one production fused-kernel launch; slot rotation/cache eviction occurs outside that event interval.\n";
        std::cout << "family,case,input_dtypes,output_dtype,input_shapes,output_shape,kernel_name,launch_kind,selected_path,elements_per_thread,packet_scalars,input_packet_bytes,output_packet_bytes,grid_x,grid_y,grid_z,block_x,block_y,block_z,registers_per_thread,local_bytes_per_thread,static_shared_bytes,device_runtime_extent,runtime_extent_source,pool_slots,pool_touched_bytes,pool_over_l2,reuse_distance_bytes,reuse_distance_over_l2,cache_control,effective_bytes,compulsory_bytes,model_logical_bytes,median_ms,best_ms,worst_ms,effective_gb_s,compulsory_gb_s\n";

        size_t selected_count = 0;
        for (const CensusCase& c : cases) {
            if (!selected(c, options)) continue;
            ++selected_count;
            runCase(c, options, static_cast<uint64_t>(l2_cache_bytes), stream, placement);
        }
        if (selected_count == 0) {
            throw std::runtime_error("No fused census cases matched the requested filters");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "FusedKernelCensus failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
