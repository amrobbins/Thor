#include "Utilities/TensorOperations/DeepLearning/CudnnRaggedSoftmax.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

[[noreturn]] void throwInvalid(const string& message) {
    throw invalid_argument("Invalid cuDNN ragged Softmax descriptor: " + message);
}

bool isSupportedIoDtype(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

uint64_t checkedMul(uint64_t a, uint64_t b, string_view what) {
    if (a != 0 && b > numeric_limits<uint64_t>::max() / a) {
        throw invalid_argument(string("cuDNN ragged Softmax ") + string(what) + " overflows uint64_t.");
    }
    return a * b;
}

int checkedInt(uint64_t value, string_view what) {
    if (value == 0) {
        throw invalid_argument(string("cuDNN ragged Softmax ") + string(what) + " must be non-zero.");
    }
    if (value > static_cast<uint64_t>(numeric_limits<int>::max())) {
        throw invalid_argument(string("cuDNN ragged Softmax ") + string(what) + " exceeds cuDNN's int dimension limit.");
    }
    return static_cast<int>(value);
}

vector<uint64_t> packedDimensions(const CudnnRaggedSoftmaxDescriptor& descriptor) {
    vector<uint64_t> dimensions;
    dimensions.reserve(descriptor.trailingDimensions.size() + 1);
    dimensions.push_back(descriptor.maxTotalValues);
    dimensions.insert(dimensions.end(), descriptor.trailingDimensions.begin(), descriptor.trailingDimensions.end());
    return dimensions;
}

void requireGpuTensor(const Tensor& tensor, int gpuNum, DataType dtype, const vector<uint64_t>& dimensions, string_view name) {
    if (!tensor.isInitialized()) {
        throw invalid_argument(string("cuDNN ragged Softmax tensor '") + string(name) + "' is not initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument(string("cuDNN ragged Softmax tensor '") + string(name) + "' must be a GPU tensor.");
    }
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument(string("cuDNN ragged Softmax tensor '") + string(name) + "' is on the wrong GPU.");
    }
    if (tensor.getDataType() != dtype) {
        throw invalid_argument(string("cuDNN ragged Softmax tensor '") + string(name) + "' has the wrong dtype.");
    }
    if (!tensor.isDenseContiguous()) {
        throw invalid_argument(string("cuDNN ragged Softmax tensor '") + string(name) + "' must be dense contiguous.");
    }
    if (tensor.getDimensions() != dimensions) {
        throw invalid_argument(string("cuDNN ragged Softmax tensor '") + string(name) + "' has the wrong packed dimensions.");
    }
}

cudnnSoftmaxAlgorithm_t algorithmFor(CudnnRaggedSoftmaxKind kind) {
    switch (kind) {
        case CudnnRaggedSoftmaxKind::Softmax:
            return CUDNN_SOFTMAX_ACCURATE;
        case CudnnRaggedSoftmaxKind::LogSoftmax:
            return CUDNN_SOFTMAX_LOG;
    }
    THOR_UNREACHABLE();
}



}  // namespace

void CudnnRaggedSoftmaxDescriptor::validate() const {
    if (maxTotalValues == 0) {
        throwInvalid("maxTotalValues must be non-zero");
    }
    if (trailingDimensions.empty()) {
        throwInvalid("at least one trailing dimension is required");
    }
    if (!isSupportedIoDtype(dataType)) {
        throwInvalid("I/O dtype must be FP16, BF16, or FP32");
    }

    uint64_t outerPerValueCount = 1;
    for (size_t i = 0; i + 1 < trailingDimensions.size(); ++i) {
        if (trailingDimensions[i] == 0) {
            throwInvalid("trailing dimensions must be non-zero");
        }
        outerPerValueCount = checkedMul(outerPerValueCount, trailingDimensions[i], "outer-per-value size");
    }
    if (trailingDimensions.back() == 0) {
        throwInvalid("trailing dimensions must be non-zero");
    }

    const uint64_t maxOuter = checkedMul(maxTotalValues, outerPerValueCount, "maximum outer size");
    (void)checkedInt(maxOuter, "maximum outer size");
    (void)checkedInt(trailingDimensions.back(), "channel count");
    (void)checkedMul(maxOuter, trailingDimensions.back(), "capacity element count");
}

uint64_t CudnnRaggedSoftmaxDescriptor::outerPerValue() const {
    validate();
    uint64_t result = 1;
    for (size_t i = 0; i + 1 < trailingDimensions.size(); ++i) {
        result = checkedMul(result, trailingDimensions[i], "outer-per-value size");
    }
    return result;
}

uint64_t CudnnRaggedSoftmaxDescriptor::channelCount() const {
    validate();
    return trailingDimensions.back();
}

uint64_t CudnnRaggedSoftmaxDescriptor::capacityElementCount() const {
    validate();
    return checkedMul(checkedMul(maxTotalValues, outerPerValue(), "capacity outer size"), channelCount(), "capacity element count");
}

CudnnRaggedSoftmaxExecutionState::CudnnRaggedSoftmaxExecutionState(CudnnRaggedSoftmaxExecutionState&& other) noexcept
    : descriptor_(std::move(other.descriptor_)), gpu_num_(other.gpu_num_), io_descriptor_(other.io_descriptor_) {
    other.gpu_num_ = -1;
    other.io_descriptor_ = nullptr;
}

CudnnRaggedSoftmaxExecutionState& CudnnRaggedSoftmaxExecutionState::operator=(CudnnRaggedSoftmaxExecutionState&& other) noexcept {
    if (this != &other) {
        release();
        descriptor_ = std::move(other.descriptor_);
        gpu_num_ = other.gpu_num_;
        io_descriptor_ = other.io_descriptor_;
        other.gpu_num_ = -1;
        other.io_descriptor_ = nullptr;
    }
    return *this;
}

CudnnRaggedSoftmaxExecutionState::~CudnnRaggedSoftmaxExecutionState() { release(); }


void CudnnRaggedSoftmaxExecutionState::configureActiveValueCount(uint64_t activeValueCount) {
    if (activeValueCount > descriptor_.maxTotalValues) {
        throw invalid_argument("cuDNN ragged Softmax activeValueCount exceeds maxTotalValues.");
    }
    if (activeValueCount == 0) {
        return;
    }

    const uint64_t activeOuter = checkedMul(activeValueCount, descriptor_.outerPerValue(), "active outer size");
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(io_descriptor_,
                                           CUDNN_TENSOR_NCHW,
                                           CudnnHelper::getCudnnDataType(descriptor_.dataType),
                                           checkedInt(activeOuter, "active outer size"),
                                           checkedInt(descriptor_.channelCount(), "channel count"),
                                           1,
                                           1));
}

void CudnnRaggedSoftmaxExecutionState::release() noexcept {
    if (io_descriptor_ != nullptr) {
        (void)cudnnDestroyTensorDescriptor(io_descriptor_);
        io_descriptor_ = nullptr;
    }
}

CudnnRaggedSoftmax& CudnnRaggedSoftmax::instance() {
    static CudnnRaggedSoftmax instance;
    return instance;
}

CudnnRaggedSoftmaxExecutionState CudnnRaggedSoftmax::prepare(const CudnnRaggedSoftmaxDescriptor& descriptor, Stream stream) const {
    descriptor.validate();
    cudnnTensorDescriptor_t ioDescriptor = nullptr;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&ioDescriptor));
    try {
        // Give the operation-local descriptor a fully valid capacity shape at
        // preparation time. Runtime only mutates dimensions to a smaller exact
        // prefix; it never allocates or prepares backend execution state.
        const uint64_t maxOuter = checkedMul(descriptor.maxTotalValues, descriptor.outerPerValue(), "maximum outer size");
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(ioDescriptor,
                                               CUDNN_TENSOR_NCHW,
                                               CudnnHelper::getCudnnDataType(descriptor.dataType),
                                               checkedInt(maxOuter, "maximum outer size"),
                                               checkedInt(descriptor.channelCount(), "channel count"),
                                               1,
                                               1));
    } catch (...) {
        (void)cudnnDestroyTensorDescriptor(ioDescriptor);
        throw;
    }
    return CudnnRaggedSoftmaxExecutionState(descriptor, stream.getGpuNum(), ioDescriptor);
}

void CudnnRaggedSoftmax::forward(CudnnRaggedSoftmaxExecutionState& state,
                                 const CudnnRaggedSoftmaxForwardArgs& args,
                                 Stream stream) const {
    const CudnnRaggedSoftmaxDescriptor& descriptor = state.descriptor();
    if (stream.getGpuNum() != state.gpuNum()) {
        throw invalid_argument("cuDNN ragged Softmax execution stream is on the wrong GPU.");
    }
    const vector<uint64_t> dimensions = packedDimensions(descriptor);
    requireGpuTensor(args.x, state.gpuNum(), descriptor.dataType, dimensions, "x");
    requireGpuTensor(args.y, state.gpuNum(), descriptor.dataType, dimensions, "y");
    if (args.activeValueCount > descriptor.maxTotalValues) {
        throw invalid_argument("cuDNN ragged Softmax activeValueCount exceeds maxTotalValues.");
    }
    if (args.activeValueCount == 0) {
        return;
    }

    state.configureActiveValueCount(args.activeValueCount);
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(stream.getCudnnHandle(),
                                    algorithmFor(descriptor.kind),
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    state.io_descriptor_,
                                    args.x.getMemPtr<void>(),
                                    &beta,
                                    state.io_descriptor_,
                                    const_cast<void*>(static_cast<const void*>(args.y.getMemPtr<void>()))));
}

void CudnnRaggedSoftmax::backward(CudnnRaggedSoftmaxExecutionState& state,
                                  const CudnnRaggedSoftmaxBackwardArgs& args,
                                  Stream stream) const {
    const CudnnRaggedSoftmaxDescriptor& descriptor = state.descriptor();
    if (stream.getGpuNum() != state.gpuNum()) {
        throw invalid_argument("cuDNN ragged Softmax execution stream is on the wrong GPU.");
    }
    const vector<uint64_t> dimensions = packedDimensions(descriptor);
    requireGpuTensor(args.y, state.gpuNum(), descriptor.dataType, dimensions, "y");
    requireGpuTensor(args.dy, state.gpuNum(), descriptor.dataType, dimensions, "dy");
    requireGpuTensor(args.dx, state.gpuNum(), descriptor.dataType, dimensions, "dx");
    if (args.activeValueCount > descriptor.maxTotalValues) {
        throw invalid_argument("cuDNN ragged Softmax activeValueCount exceeds maxTotalValues.");
    }
    if (args.activeValueCount == 0) {
        return;
    }

    state.configureActiveValueCount(args.activeValueCount);
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxBackward(stream.getCudnnHandle(),
                                     algorithmFor(descriptor.kind),
                                     CUDNN_SOFTMAX_MODE_CHANNEL,
                                     &alpha,
                                     state.io_descriptor_,
                                     args.y.getMemPtr<void>(),
                                     state.io_descriptor_,
                                     args.dy.getMemPtr<void>(),
                                     &beta,
                                     state.io_descriptor_,
                                     const_cast<void*>(static_cast<const void*>(args.dx.getMemPtr<void>()))));
}
