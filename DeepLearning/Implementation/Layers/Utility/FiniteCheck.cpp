#include "DeepLearning/Implementation/Layers/Utility/FiniteCheck.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace {

template <typename T>
double finiteCheckCpuToDouble(T value) {
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_fp8_e4m3> ||
                  std::is_same_v<T, __nv_fp8_e5m2>) {
        return static_cast<double>(static_cast<float>(value));
    } else {
        return static_cast<double>(value);
    }
}

template <typename T>
FiniteCheckResult checkCpuTyped(const T *data, uint64_t numElements, uint32_t maxReportedIndices) {
    FiniteCheckResult result{};
    for (uint64_t index = 0; index < numElements; ++index) {
        const double value = finiteCheckCpuToDouble(data[index]);
        FiniteCheckSampleKind kind = FiniteCheckSampleKind::NONE;
        if (std::isnan(value)) {
            ++result.nanCount;
            kind = FiniteCheckSampleKind::NAN_VALUE;
        } else if (std::isinf(value)) {
            if (std::signbit(value)) {
                ++result.negativeInfinityCount;
                kind = FiniteCheckSampleKind::NEGATIVE_INFINITY;
            } else {
                ++result.positiveInfinityCount;
                kind = FiniteCheckSampleKind::POSITIVE_INFINITY;
            }
        }

        if (kind == FiniteCheckSampleKind::NONE)
            continue;

        const uint64_t sample = result.totalNonFinite++;
        if (sample < maxReportedIndices) {
            result.flatIndices[sample] = index;
            result.kinds[sample] = static_cast<uint32_t>(kind);
        }
    }
    return result;
}

std::string dimensionsString(const std::vector<uint64_t> &dimensions) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0)
            out << ", ";
        out << dimensions[i];
    }
    out << ']';
    return out.str();
}

std::mutex finiteCheckReportMutex;

std::string sampleKindString(uint32_t kind) {
    switch (static_cast<FiniteCheckSampleKind>(kind)) {
        case FiniteCheckSampleKind::NAN_VALUE:
            return "NaN";
        case FiniteCheckSampleKind::POSITIVE_INFINITY:
            return "+Inf";
        case FiniteCheckSampleKind::NEGATIVE_INFINITY:
            return "-Inf";
        default:
            return "unknown";
    }
}

}  // namespace

FiniteCheck::FiniteCheck(std::string tensorLabel,
                         uint64_t apiTensorId,
                         uint64_t originalApiTensorId,
                         bool checkForward,
                         bool checkBackward,
                         bool failOnNonFinite,
                         uint32_t maxReportedIndices)
    : tensorLabel(std::move(tensorLabel)),
      apiTensorId(apiTensorId),
      originalApiTensorId(originalApiTensorId),
      checkForward(checkForward),
      checkBackward(checkBackward),
      failOnNonFinite(failOnNonFinite),
      maxReportedIndices(maxReportedIndices) {
    if (!checkForward && !checkBackward)
        throw std::invalid_argument("FiniteCheck must check forward, backward, or both.");
    if (maxReportedIndices > FINITE_CHECK_MAX_REPORTED_INDICES) {
        throw std::invalid_argument("FiniteCheck maxReportedIndices exceeds the supported maximum of " +
                                    std::to_string(FINITE_CHECK_MAX_REPORTED_INDICES) + ".");
    }
}

FiniteCheck::~FiniteCheck() {
    if (gpuResult != nullptr) {
        try {
            cleanup();
        } catch (...) {
        }
    }
}

std::optional<Tensor> FiniteCheck::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return featureInput.value();
}

void FiniteCheck::connectToNextLayer(Layer *nextLayer, int driverConnectionType, int loaderConnectionType) {
    Layer::connectToNextLayer(nextLayer, driverConnectionType, loaderConnectionType);
    fuseBackwardAlias();
}

void FiniteCheck::fuseBackwardAlias() {
    if (!errorInput.has_value() || !errorOutput.has_value())
        return;

    THOR_THROW_IF_FALSE(errorInput.value().getDescriptor() == errorOutput.value().getDescriptor());
    if (previousLayer.has_value())
        previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
    errorOutput = errorInput;
}

void FiniteCheck::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.has_value());

    if (TensorDescriptor::isIntegralType(featureInput.value().getDataType()))
        return;
    if (featureInput.value().getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU)
        return;

    ScopedGpu scopedGpu(featureInput.value().getPlacement().getDeviceNum());
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpuResult), sizeof(FiniteCheckResult)));
}

void FiniteCheck::cleanup() {
    if (gpuResult != nullptr) {
        const int gpuNum = featureInput.has_value() &&
                                   featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU
                               ? featureInput.value().getPlacement().getDeviceNum()
                               : stream.getGpuNum();
        ScopedGpu scopedGpu(gpuNum);
        CUDA_CHECK(cudaFree(gpuResult));
        gpuResult = nullptr;
    }
    Layer::cleanup();
}

void FiniteCheck::infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) {
    (void)outputTensor;
    if (checkForward && inputTensor.has_value())
        checkTensor(inputTensor.value(), "forward", "activation", stream);
}

void FiniteCheck::backProp(std::optional<Tensor> dataIn,
                           std::optional<Tensor> errorIn,
                           std::optional<Tensor> errorOut,
                           Stream stream) {
    (void)dataIn;
    (void)errorOut;
    if (checkBackward && errorIn.has_value())
        checkTensor(errorIn.value(), "backward", "incoming_gradient", stream);
}

void FiniteCheck::checkTensor(const Tensor &tensor, const char *direction, const char *tensorRole, Stream stream) {
    if (TensorDescriptor::isIntegralType(tensor.getDataType()) || tensor.getTotalNumElements() == 0)
        return;

    // A placed network may be submitted from more than one host thread. The
    // diagnostic workspace and host result are intentionally one-at-a-time.
    std::unique_lock<std::mutex> checkLock(mtx);
    if (!tensor.isDenseContiguous()) {
        throw std::runtime_error("FiniteCheck requires a dense contiguous tensor. label=\"" +
                                 (tensorLabel.empty() ? std::string("<unnamed>") : tensorLabel) + "\" direction=" + direction +
                                 " dtype=" + TensorDescriptor::getElementTypeName(tensor.getDataType()));
    }

    FiniteCheckResult result{};
    if (tensor.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        // CPU layer kernels may be queued as CUDA host functions on this stream.
        // Wait before directly reading the aliased host tensor.
        stream.synchronize();
        result = checkCpuTensor(tensor);
    } else {
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream.getStream(), &captureStatus));
        if (captureStatus != cudaStreamCaptureStatusNone) {
            throw std::runtime_error("FiniteCheck cannot execute during CUDA graph capture. Disable graph capture while using diagnostic "
                                     "FiniteCheck layers.");
        }
        THOR_THROW_IF_FALSE(gpuResult != nullptr);
        ScopedGpu scopedGpu(tensor.getPlacement().getDeviceNum());
        CUDA_CHECK(cudaMemsetAsync(gpuResult, 0, sizeof(FiniteCheckResult), stream.getStream()));
        launchFiniteCheck(
            tensor.getMemPtr(), tensor.getDataType(), tensor.getTotalNumElements(), maxReportedIndices, gpuResult, stream);
        CUDA_CHECK(cudaMemcpyAsync(&result, gpuResult, sizeof(FiniteCheckResult), cudaMemcpyDeviceToHost, stream.getStream()));

        // FiniteCheck is deliberately a debugging barrier. A host-visible report and a
        // synchronous exception require knowing the result before downstream work is submitted.
        stream.synchronize();
    }

    if (result.totalNonFinite == 0)
        return;

    const std::string message = formatFailure(tensor, direction, tensorRole, result);
    if (failOnNonFinite)
        throw std::runtime_error(message);

    std::lock_guard<std::mutex> lock(finiteCheckReportMutex);
    std::cerr << message << std::endl;
}

FiniteCheckResult FiniteCheck::checkCpuTensor(const Tensor &tensor) const {
    switch (tensor.getDataType()) {
        case DataType::FP8_E4M3:
            return checkCpuTyped(static_cast<const __nv_fp8_e4m3 *>(tensor.getMemPtr()), tensor.getTotalNumElements(), maxReportedIndices);
        case DataType::FP8_E5M2:
            return checkCpuTyped(static_cast<const __nv_fp8_e5m2 *>(tensor.getMemPtr()), tensor.getTotalNumElements(), maxReportedIndices);
        case DataType::FP16:
            return checkCpuTyped(static_cast<const half *>(tensor.getMemPtr()), tensor.getTotalNumElements(), maxReportedIndices);
        case DataType::BF16:
            return checkCpuTyped(static_cast<const __nv_bfloat16 *>(tensor.getMemPtr()), tensor.getTotalNumElements(), maxReportedIndices);
        case DataType::FP32:
            return checkCpuTyped(static_cast<const float *>(tensor.getMemPtr()), tensor.getTotalNumElements(), maxReportedIndices);
        case DataType::FP64:
            return checkCpuTyped(static_cast<const double *>(tensor.getMemPtr()), tensor.getTotalNumElements(), maxReportedIndices);
        default:
            throw std::invalid_argument("FiniteCheck CPU scan only accepts floating-point tensor storage types.");
    }
}

std::string FiniteCheck::formatFailure(const Tensor &tensor,
                                       const char *direction,
                                       const char *tensorRole,
                                       const FiniteCheckResult &result) const {
    const TensorDescriptor descriptor = tensor.getDescriptor();
    std::ostringstream out;
    out << "FiniteCheck detected non-finite values"
        << ": label=\"" << (tensorLabel.empty() ? "<unnamed>" : tensorLabel) << '\"'
        << " finite_check_layer_id=" << getId() << " direction=" << direction << " tensor_role=" << tensorRole
        << " api_tensor_id=" << apiTensorId << " original_api_tensor_id=" << originalApiTensorId
        << " physical_tensor_id=" << tensor.getTensorId() << " dtype=" << TensorDescriptor::getElementTypeName(tensor.getDataType())
        << " shape=" << dimensionsString(tensor.getDimensions()) << " elements=" << tensor.getTotalNumElements()
        << " non_finite=" << result.totalNonFinite << " nan=" << result.nanCount
        << " positive_infinity=" << result.positiveInfinityCount << " negative_infinity=" << result.negativeInfinityCount;

    const uint64_t reported = std::min<uint64_t>(result.totalNonFinite, maxReportedIndices);
    if (reported != 0) {
        out << " samples=[";
        for (uint64_t i = 0; i < reported; ++i) {
            if (i != 0)
                out << ", ";
            const uint64_t flatIndex = result.flatIndices[i];
            out << "{flat_index=" << flatIndex << ", index=" << dimensionsString(descriptor.getDimensionalIndex(flatIndex))
                << ", value=" << sampleKindString(result.kinds[i]) << '}';
        }
        out << ']';
    }

    if (!failOnNonFinite)
        out << " action=report_only";
    return out.str();
}

}  // namespace ThorImplementation
