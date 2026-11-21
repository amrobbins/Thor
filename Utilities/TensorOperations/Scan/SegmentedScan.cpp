#include "Utilities/TensorOperations/Scan/SegmentedScan.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

namespace {

namespace K = SegmentedScanKernels;

bool isFloat(DataType dtype) {
    return dtype == DataType::FP32 || dtype == DataType::FP16 || dtype == DataType::BF16 ||
           dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
}

bool isOwnDataTypeSupported(DataType dtype) { return dtype == DataType::UINT32 || isFloat(dtype); }

std::string name(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

const char* name(SegmentedScanOp op) {
    switch (op) {
        case SegmentedScanOp::Sum:
            return "sum";
        case SegmentedScanOp::Max:
            return "max";
        case SegmentedScanOp::Min:
            return "min";
        case SegmentedScanOp::Product:
            return "product";
        case SegmentedScanOp::ArgMax:
            return "argmax";
        case SegmentedScanOp::ArgMin:
            return "argmin";
    }
    return "unknown";
}

// Float sums and products may write any float dtype (their FP32 accumulator rounded once into it); min and max write
// their own dtype, and the arg scans UINT32 indices.
constexpr bool converts(SegmentedScanOp op) { return op == SegmentedScanOp::Sum || op == SegmentedScanOp::Product; }

// An arg scan's key holds a 31-bit index (SegmentedScanKernel.cuh, keyOf): at most 2^31 elements, CUB's cap too.
constexpr uint64_t kMaxArgElements = uint64_t{1} << 31;

// Calls fn.template operator()<T>() for the element type of a supported dtype.
template <typename Fn>
void dispatchType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::FP32:
            return fn.template operator()<float>();
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
        case DataType::FP16:
            return fn.template operator()<__half>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP8_E4M3:
            return fn.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return fn.template operator()<__nv_fp8_e5m2>();
        default:
            throw std::invalid_argument("SegmentedScan: unsupported dtype " + name(dtype) + ".");
    }
}

template <typename T>
constexpr bool kIsFloat = std::is_same_v<T, float> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> ||
                          std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;

// The type a kernel writes for an output dtype: a float sum or product of 2 or 1 bytes writes the SharedFloat of its
// size (one kernel for FP16 and BF16, one for E4M3 and E5M2; the launch picks the format); every other scan its type.
template <typename T, typename O, SegmentedScanOp OP>
using Stored = std::conditional_t<converts(OP) && kIsFloat<T> && kIsFloat<O> && sizeof(O) < 4,
                                  K::SharedFloat<static_cast<uint32_t>(sizeof(O))>,
                                  O>;

// The kernels instantiated in SegmentedScanFp32.cu, SegmentedScanF16.cu, SegmentedScanFp8.cu and SegmentedScanArg.cu,
// by the dtypes' types: the same rule as isSupported.
template <typename T, typename O, SegmentedScanOp OP>
constexpr bool kInstantiated =
    isArg(OP) ? std::is_same_v<O, uint32_t>
              : std::is_same_v<T, O> || (converts(OP) && kIsFloat<T> && kIsFloat<O>);

// Which kernel of a (T, O, OP) a launch takes.
struct Variant {
    SegmentedScanMode mode;
    SegmentedScanDirection direction;
    bool ragged;
    DataType offset_dtype;  // ragged rows
};

template <typename T, typename O, SegmentedScanOp OP, SegmentedScanMode MODE, SegmentedScanDirection DIR>
void launchRows(const Variant& variant, const K::Launch& args, cudaStream_t stream) {
    if (!variant.ragged) return K::launch<T, O, OP, MODE, DIR, void>(args, stream);
    if (variant.offset_dtype == DataType::UINT64) return K::launch<T, O, OP, MODE, DIR, uint64_t>(args, stream);
    return K::launch<T, O, OP, MODE, DIR, uint32_t>(args, stream);
}

template <typename T, typename O, SegmentedScanOp OP, SegmentedScanMode MODE>
void launchDirection(const Variant& variant, const K::Launch& args, cudaStream_t stream) {
    if (variant.direction == SegmentedScanDirection::Reverse) {
        return launchRows<T, O, OP, MODE, SegmentedScanDirection::Reverse>(variant, args, stream);
    }
    return launchRows<T, O, OP, MODE, SegmentedScanDirection::Forward>(variant, args, stream);
}

template <typename T, typename O, SegmentedScanOp OP>
void launchFor(const Variant& variant, const K::Launch& args, cudaStream_t stream) {
    if constexpr (kInstantiated<T, O, OP>) {
        using S = Stored<T, O, OP>;
        if (variant.mode == SegmentedScanMode::Exclusive) {
            return launchDirection<T, S, OP, SegmentedScanMode::Exclusive>(variant, args, stream);
        }
        return launchDirection<T, S, OP, SegmentedScanMode::Inclusive>(variant, args, stream);
    } else {
        throw std::logic_error("SegmentedScan: no kernel for a supported dtype pair.");
    }
}

}  // namespace

bool SegmentedScan::isSupported(SegmentedScanOp op, DataType input_dtype, DataType output_dtype) {
    if (!isOwnDataTypeSupported(input_dtype)) return false;
    if (isArg(op)) return output_dtype == DataType::UINT32;
    if (input_dtype == output_dtype) return true;
    return converts(op) && isFloat(input_dtype) && isFloat(output_dtype);
}

bool SegmentedScan::isSupported(SegmentedScanOp op, const Tensor& input, const Tensor& output) {
    if (!input.isInitialized() || !output.isInitialized()) return false;
    if (!isSupported(op, input.getDataType(), output.getDataType())) return false;
    if (input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) return false;
    if (input.getPlacement() != output.getPlacement()) return false;
    if (input.getTotalNumElements() != output.getTotalNumElements()) return false;
    if (isArg(op) && input.getTotalNumElements() > kMaxArgElements) return false;
    return input.isAligned(16) && output.isAligned(16);
}

bool SegmentedScan::isValueOutputSupported(SegmentedScanOp op, const Tensor& input, const Tensor& arg_values) {
    if (!isArg(op) || !input.isInitialized() || !arg_values.isInitialized()) return false;
    return arg_values.getDataType() == input.getDataType() && arg_values.getPlacement() == input.getPlacement() &&
           arg_values.getTotalNumElements() == input.getTotalNumElements() && arg_values.isAligned(16);
}

bool SegmentedScan::isOffsetDataTypeSupported(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

SegmentedScan::SegmentedScan(SegmentedScanOp op,
                             SegmentedScanMode mode,
                             SegmentedScanDirection direction,
                             const Tensor& input,
                             const Tensor& output,
                             const std::optional<Tensor>& arg_values)
    : op(op),
      mode(mode),
      direction(direction),
      input(input),
      output(output),
      arg_values(arg_values.value_or(Tensor())) {
    if (!input.isInitialized() || !output.isInitialized()) {
        throw std::invalid_argument("SegmentedScan: input and output must be initialized.");
    }
    if (!isSupported(op, input.getDataType(), output.getDataType())) {
        throw std::invalid_argument("SegmentedScan: no " + std::string(name(op)) + " scan from " +
                                    name(input.getDataType()) + " to " + name(output.getDataType()) + ".");
    }
    if (input.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        input.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument("SegmentedScan: input and output must be on the same GPU.");
    }
    if (input.getTotalNumElements() != output.getTotalNumElements()) {
        throw std::invalid_argument("SegmentedScan: input and output must have the same element count.");
    }
    if (isArg(op) && input.getTotalNumElements() > kMaxArgElements) {
        throw std::invalid_argument("SegmentedScan: an " + std::string(name(op)) + " scan takes at most 2^31 " +
                                    "elements, not " + std::to_string(input.getTotalNumElements()) + ".");
    }
    if (arg_values.has_value() && !isValueOutputSupported(op, input, arg_values.value())) {
        throw std::invalid_argument(
            "SegmentedScan: values are written only by an arg scan, to a 16-byte aligned tensor of the input's dtype, "
            "GPU and element count.");
    }
    if (!input.isAligned(16) || !output.isAligned(16)) {
        throw std::invalid_argument("SegmentedScan: input and output must start on 16-byte aligned addresses.");
    }
    num_items = input.getTotalNumElements();
    const uint64_t input_bytes = TensorDescriptor::getElementSizeInBytes(input.getDataType());
    const uint64_t output_bytes = TensorDescriptor::getElementSizeInBytes(output.getDataType());
    tile_elements = K::tileElements(static_cast<uint32_t>(input_bytes), static_cast<uint32_t>(output_bytes));
    num_tiles = (num_items + tile_elements - 1) / tile_elements;
    if (num_tiles > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("SegmentedScan: " + std::to_string(num_items) + " elements need more tiles (" +
                                    std::to_string(num_tiles) + ") than a grid holds.");
    }
}

SegmentedScan::SegmentedScan(SegmentedScanOp op,
                             SegmentedScanMode mode,
                             SegmentedScanDirection direction,
                             const Tensor& input,
                             const Tensor& output,
                             uint64_t row_length,
                             const std::optional<Tensor>& arg_values)
    : SegmentedScan(op, mode, direction, input, output, arg_values) {
    if (num_items != 0 && (row_length == 0 || num_items % row_length != 0)) {
        throw std::invalid_argument("SegmentedScan: row length " + std::to_string(row_length) +
                                    " does not divide the element count " + std::to_string(num_items) + ".");
    }
    this->row_length = row_length;
    // Every tile starts a row when the row length divides the tile: no tile publishes or waits (the kernel decides the
    // same way).
    if (num_items != 0 && tile_elements % row_length != 0) makeWorkspace();
}

SegmentedScan::SegmentedScan(SegmentedScanOp op,
                             SegmentedScanMode mode,
                             SegmentedScanDirection direction,
                             const Tensor& input,
                             const Tensor& output,
                             const Tensor& segment_offsets,
                             const std::optional<Tensor>& arg_values)
    : SegmentedScan(op, mode, direction, input, output, arg_values) {
    if (!segment_offsets.isInitialized() || !isOffsetDataTypeSupported(segment_offsets.getDataType())) {
        throw std::invalid_argument("SegmentedScan: segment offsets must be UINT32 or UINT64.");
    }
    if (segment_offsets.getPlacement() != input.getPlacement()) {
        throw std::invalid_argument("SegmentedScan: segment offsets must be on the input's GPU.");
    }
    const std::vector<uint64_t> dims = segment_offsets.getDimensions();
    if (dims.size() != 1 || dims[0] == 0) {
        throw std::invalid_argument("SegmentedScan: segment offsets must be a non-empty rank-1 tensor [rows + 1].");
    }
    if (dims[0] - 1 >= std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("SegmentedScan: " + std::to_string(dims[0] - 1) + " rows are more than 2^32 - 2.");
    }
    this->segment_offsets = segment_offsets;
    rows = static_cast<uint32_t>(dims[0] - 1);
    if (num_items != 0) makeWorkspace();
}

// The workspace is cleared here, once, and never again: every launch writes every run word, and takes its epoch from
// the words the last launch left (SegmentedScanKernel.cuh). Nothing on the host changes from launch to
// launch, so a launch captured in a CUDA graph replays correctly, before or after launches outside the graph.
void SegmentedScan::makeWorkspace() {
    workspace = Tensor(input.getPlacement(),
                       TensorDescriptor(DataType::UINT64, {K::workspaceWords(op, num_items, tile_elements)}));
    Stream clear_stream(input.getPlacement().getDeviceNum());
    workspace.memsetAsync(clear_stream, 0);
    clear_stream.synchronize();
    auto* words = reinterpret_cast<unsigned long long*>(workspace.getMemPtr<uint64_t>());
    const uint64_t groups = (num_tiles + K::kGroup - 1) / K::kGroup;
    const uint64_t run_words = K::runWords(op);
    workspace_args.tile_runs = words;
    workspace_args.group_runs = words + run_words * num_tiles;
    workspace_args.super_runs = words + run_words * (num_tiles + groups);
}

uint64_t SegmentedScan::getWorkspaceSizeInBytes() const {
    return workspace.isInitialized() ? workspace.getArraySizeInBytes() : 0;
}

void SegmentedScan::runOn(Stream& stream) const {
    if (num_items == 0) return;
    ScopedGpu scoped_gpu(input.getPlacement().getDeviceNum());
    const bool ragged = segment_offsets.isInitialized();
    K::Launch args;
    args.input = input.getMemPtr<void>();
    args.output = output.getMemPtr<void>();
    args.n = num_items;
    args.columns = K::columnsOf(ragged ? 0 : row_length);
    args.offsets = ragged ? segment_offsets.getMemPtr<void>() : nullptr;
    args.rows = rows;
    args.workspace = workspace_args;
    args.workspace.num_tiles = num_tiles;
    args.arg_values = arg_values.isInitialized() ? arg_values.getMemPtr<void>() : nullptr;
    args.alternate_output = output.getDataType() == DataType::BF16 || output.getDataType() == DataType::FP8_E5M2;

    const Variant variant{mode, direction, ragged, ragged ? segment_offsets.getDataType() : DataType::UINT32};
    const cudaStream_t cuda_stream = stream.getStream();
    dispatchType(input.getDataType(), [&]<typename T>() {
        dispatchType(output.getDataType(), [&]<typename O>() {
            switch (op) {
                case SegmentedScanOp::Sum:
                    return launchFor<T, O, SegmentedScanOp::Sum>(variant, args, cuda_stream);
                case SegmentedScanOp::Max:
                    return launchFor<T, O, SegmentedScanOp::Max>(variant, args, cuda_stream);
                case SegmentedScanOp::Min:
                    return launchFor<T, O, SegmentedScanOp::Min>(variant, args, cuda_stream);
                case SegmentedScanOp::Product:
                    return launchFor<T, O, SegmentedScanOp::Product>(variant, args, cuda_stream);
                case SegmentedScanOp::ArgMax:
                    return launchFor<T, O, SegmentedScanOp::ArgMax>(variant, args, cuda_stream);
                case SegmentedScanOp::ArgMin:
                    return launchFor<T, O, SegmentedScanOp::ArgMin>(variant, args, cuda_stream);
            }
        });
    });
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ThorImplementation
