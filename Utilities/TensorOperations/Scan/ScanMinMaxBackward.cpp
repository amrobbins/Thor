#include "Utilities/TensorOperations/Scan/ScanMinMaxBackward.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

namespace {

namespace K = ScanMinMaxBackwardKernels;

bool isFloat(DataType dtype) {
    return dtype == DataType::FP32 || dtype == DataType::FP16 || dtype == DataType::BF16 ||
           dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
}

K::FloatFormat formatOf(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return K::FloatFormat::Fp32;
        case DataType::FP16:
            return K::FloatFormat::Fp16;
        case DataType::BF16:
            return K::FloatFormat::Bf16;
        case DataType::FP8_E4M3:
            return K::FloatFormat::E4m3;
        default:
            return K::FloatFormat::E5m2;
    }
}

std::string name(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

// A key holds a 31-bit index (winnerKey), as the arg scans' do.
constexpr uint64_t kMaxElements = uint64_t{1} << 31;

}  // namespace

bool ScanMinMaxBackward::isSupported(DataType x_dtype, DataType dy_dtype, DataType dx_dtype) {
    return isFloat(x_dtype) && isFloat(dy_dtype) && isFloat(dx_dtype);
}

ScanMinMaxBackward::ScanMinMaxBackward(bool min,
                                       SegmentedScanMode mode,
                                       SegmentedScanDirection direction,
                                       const Tensor& x,
                                       const Tensor& dy,
                                       const Tensor& dx)
    : direction(direction), x(x), dy(dy), dx(dx) {
    if (!x.isInitialized() || !dy.isInitialized() || !dx.isInitialized()) {
        throw std::invalid_argument("ScanMinMaxBackward: x, dy and dx must be initialized.");
    }
    if (!isSupported(x.getDataType(), dy.getDataType(), dx.getDataType())) {
        throw std::invalid_argument("ScanMinMaxBackward: x, dy and dx must be floats, not " + name(x.getDataType()) + ", " +
                                    name(dy.getDataType()) + " and " + name(dx.getDataType()) + ".");
    }
    if (x.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU || x.getPlacement() != dy.getPlacement() ||
        x.getPlacement() != dx.getPlacement()) {
        throw std::invalid_argument("ScanMinMaxBackward: x, dy and dx must be on the same GPU.");
    }
    const uint64_t n = x.getTotalNumElements();
    if (dy.getTotalNumElements() != n || dx.getTotalNumElements() != n) {
        throw std::invalid_argument("ScanMinMaxBackward: x, dy and dx must have the same element count.");
    }
    if (n > kMaxElements) {
        throw std::invalid_argument("ScanMinMaxBackward: takes at most 2^31 elements, not " + std::to_string(n) + ".");
    }
    // A chunk of 8 elements is read and written as one access (ScanMinMaxBackwardKernel.cuh, loadRawChunk): 32 bytes
    // in FP32, 16 in FP16 and BF16, 8 in FP8.
    for (const Tensor* tensor : {&x, &dy, &dx}) {
        const uint64_t chunk_bytes = K::kGradientItems * TensorDescriptor::getElementSizeInBytes(tensor->getDataType());
        if (!tensor->isAligned(chunk_bytes)) {
            throw std::invalid_argument("ScanMinMaxBackward: " + name(tensor->getDataType()) + " x, dy or dx must start on a " +
                                        std::to_string(chunk_bytes) + "-byte aligned address.");
        }
    }
    const K::WorkspaceCounts counts = K::workspaceCounts(n, K::kGradientTile);
    if (counts.tiles > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("ScanMinMaxBackward: " + std::to_string(n) + " elements need more tiles than a grid holds.");
    }
    launch_args.x_format = formatOf(x.getDataType());
    launch_args.dy_format = formatOf(dy.getDataType());
    launch_args.dx_format = formatOf(dx.getDataType());
    launch_args.n = n;
    launch_args.keys.num_tiles = counts.tiles;
    launch_args.min = min;
    launch_args.exclusive = mode == SegmentedScanMode::Exclusive;
}

ScanMinMaxBackward::ScanMinMaxBackward(bool min,
                                       SegmentedScanMode mode,
                                       SegmentedScanDirection direction,
                                       const Tensor& x,
                                       const Tensor& dy,
                                       const Tensor& dx,
                                       uint64_t row_length)
    : ScanMinMaxBackward(min, mode, direction, x, dy, dx) {
    const uint64_t n = launch_args.n;
    if (n != 0 && (row_length == 0 || n % row_length != 0)) {
        throw std::invalid_argument("ScanMinMaxBackward: row length " + std::to_string(row_length) +
                                    " does not divide the element count " + std::to_string(n) + ".");
    }
    launch_args.columns = SegmentedScanKernels::columnsOf(row_length);
    // Every tile starts a row when the row length divides the tile: no tile publishes or waits (the kernel decides the
    // same way).
    if (n != 0 && K::kGradientTile % row_length != 0) makeWorkspace();
}

ScanMinMaxBackward::ScanMinMaxBackward(bool min,
                                       SegmentedScanMode mode,
                                       SegmentedScanDirection direction,
                                       const Tensor& x,
                                       const Tensor& dy,
                                       const Tensor& dx,
                                       const Tensor& segment_offsets)
    : ScanMinMaxBackward(min, mode, direction, x, dy, dx) {
    if (!segment_offsets.isInitialized() ||
        (segment_offsets.getDataType() != DataType::UINT32 && segment_offsets.getDataType() != DataType::UINT64)) {
        throw std::invalid_argument("ScanMinMaxBackward: segment offsets must be UINT32 or UINT64.");
    }
    if (segment_offsets.getPlacement() != x.getPlacement()) {
        throw std::invalid_argument("ScanMinMaxBackward: segment offsets must be on x's GPU.");
    }
    const std::vector<uint64_t> dims = segment_offsets.getDimensions();
    if (dims.size() != 1 || dims[0] == 0) {
        throw std::invalid_argument("ScanMinMaxBackward: segment offsets must be a non-empty rank-1 tensor [rows + 1].");
    }
    if (dims[0] - 1 >= std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("ScanMinMaxBackward: " + std::to_string(dims[0] - 1) + " rows are more than 2^32 - 2.");
    }
    this->segment_offsets = segment_offsets;
    launch_args.columns = SegmentedScanKernels::columnsOf(0);
    launch_args.rows = static_cast<uint32_t>(dims[0] - 1);
    if (launch_args.n != 0) makeWorkspace();
}

// The workspace is cleared here, once, and never again. Each launch takes its epoch from the words the last launch
// left, every tile reading its own (SegmentedScanKernel.cuh, packRun): the tiles agree on the epoch only if every word
// starts equal, and a waiting tile tells this launch's words from stale ones only by it. Every launch writes every word,
// so they stay equal. Nothing on the host changes from launch to launch, so a launch captured in a CUDA graph replays
// correctly, before or after launches outside it.
void ScanMinMaxBackward::makeWorkspace() {
    const K::WorkspaceCounts counts = K::workspaceCounts(launch_args.n, K::kGradientTile);
    workspace = Tensor(x.getPlacement(), TensorDescriptor(DataType::UINT64, {counts.words()}));
    Stream clear_stream(x.getPlacement().getDeviceNum());
    workspace.memsetAsync(clear_stream, 0);
    clear_stream.synchronize();
    K::layoutWorkspace(reinterpret_cast<unsigned long long*>(workspace.getMemPtr<uint64_t>()), counts, launch_args);
}

uint64_t ScanMinMaxBackward::getWorkspaceSizeInBytes() const {
    return workspace.isInitialized() ? workspace.getArraySizeInBytes() : 0;
}

void ScanMinMaxBackward::runOn(Stream& stream) const {
    if (launch_args.n == 0) return;
    ScopedGpu scoped_gpu(x.getPlacement().getDeviceNum());
    K::BackwardLaunch args = launch_args;
    args.x = x.getMemPtr<void>();
    args.dy = dy.getMemPtr<void>();
    args.dx = dx.getMemPtr<void>();
    const bool ragged = segment_offsets.isInitialized();
    args.offsets = ragged ? segment_offsets.getMemPtr<void>() : nullptr;
    const uint32_t offset_bytes = !ragged ? 0U : segment_offsets.getDataType() == DataType::UINT64 ? 8U : 4U;
    K::launch(args, direction == SegmentedScanDirection::Reverse, offset_bytes, stream.getStream());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ThorImplementation
