#include "DeepLearning/Implementation/Layers/Utility/FiniteCheck.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
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
#include <limits>
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
    result.checkedElements = numElements;
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
                         uint32_t maxReportedIndices,
                         bool enabled,
                         std::optional<RaggedConfiguration> raggedConfiguration)
    : tensorLabel(std::move(tensorLabel)),
      apiTensorId(apiTensorId),
      originalApiTensorId(originalApiTensorId),
      checkForward(checkForward),
      enabled(enabled),
      checkBackward(checkBackward),
      failOnNonFinite(failOnNonFinite),
      maxReportedIndices(maxReportedIndices),
      raggedConfiguration(raggedConfiguration) {
    if (!checkForward && !checkBackward)
        throw std::invalid_argument("FiniteCheck must check forward, backward, or both.");
    if (maxReportedIndices > FINITE_CHECK_MAX_REPORTED_INDICES) {
        throw std::invalid_argument("FiniteCheck maxReportedIndices exceeds the supported maximum of " +
                                    std::to_string(FINITE_CHECK_MAX_REPORTED_INDICES) + ".");
    }
    if (this->raggedConfiguration.has_value()) {
        const RaggedConfiguration& config = this->raggedConfiguration.value();
        if (config.batchSize == 0 || config.maxTotalValues == 0 || config.elementsPerValue == 0 ||
            !RowPartitionDescriptor::isValidOffsetsDataType(config.offsetsDataType)) {
            throw std::invalid_argument("Ragged FiniteCheck configuration is invalid.");
        }
        if (config.maxTotalValues > std::numeric_limits<uint64_t>::max() / config.elementsPerValue) {
            throw std::overflow_error("Ragged FiniteCheck packed element capacity overflows uint64_t.");
        }
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

std::optional<Tensor> FiniteCheck::connectToPreviousLayer(
    Layer *previousLayer,
    std::optional<Tensor> connectedInput,
    Stream connectedStream,
    bool backPropagateError,
    int connectionType) {
    if (!raggedConfiguration.has_value()) {
        THOR_THROW_IF_FALSE(connectionType == 0);
        return Layer::connectToPreviousLayer(previousLayer, connectedInput, connectedStream, backPropagateError, connectionType);
    }

    if (connectionType == 0) {
        std::optional<Tensor> result =
            Layer::connectToPreviousLayer(previousLayer, connectedInput, connectedStream, backPropagateError, connectionType);
        if (rowPartitionInput.has_value()) validateRaggedInputs();
        return result;
    }
    if (connectionType != 1)
        throw std::runtime_error("Ragged FiniteCheck received an unknown physical input port.");

    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(previousLayer != nullptr);
    THOR_THROW_IF_FALSE(connectedInput.has_value());
    THOR_THROW_IF_FALSE(!rowPartitionInput.has_value());
    rowPartitionInput = connectedInput.value();
    rowPartitionStream = connectedStream;
    if (featureInput.has_value()) validateRaggedInputs();
    // Canonical offsets are a structural forward dependency. They never receive gradients.
    return std::nullopt;
}

void FiniteCheck::connectToNextLayer(Layer *nextLayer, int driverConnectionType, int loaderConnectionType) {
    Layer::connectToNextLayer(nextLayer, driverConnectionType, loaderConnectionType);
    fuseBackwardAlias();
}

void FiniteCheck::forward(std::optional<Tensor> arrivingInput, bool validationPass, uint32_t batchSize) {
    if (!raggedConfiguration.has_value()) {
        Layer::forward(arrivingInput, validationPass, batchSize);
        return;
    }

    THOR_THROW_IF_FALSE(arrivingInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(rowPartitionInput.has_value());
    const bool valuesArrival = arrivingInput.value() == featureInput.value();
    const bool partitionArrival = arrivingInput.value() == rowPartitionInput.value();
    THOR_THROW_IF_FALSE(valuesArrival || partitionArrival);

    // Disabled diagnostics are a zero-cost values identity. The structural edge
    // remains present in the graph but does not need to delay values execution.
    if (!enabled) {
        if (partitionArrival) return;
        Layer::forward(featureInput, validationPass, batchSize);
        return;
    }

    if (!pendingRaggedValidationPass.has_value()) {
        pendingRaggedValidationPass = validationPass;
        pendingRaggedBatchSize = batchSize;
    } else {
        THOR_THROW_IF_FALSE(pendingRaggedValidationPass.value() == validationPass);
        THOR_THROW_IF_FALSE(pendingRaggedBatchSize.value() == batchSize);
    }
    if (valuesArrival) raggedValuesArrived = true;
    if (partitionArrival) raggedPartitionArrived = true;
    if (!raggedValuesArrived || !raggedPartitionArrived) return;

    const bool resolvedValidationPass = pendingRaggedValidationPass.value();
    const uint32_t resolvedBatchSize = pendingRaggedBatchSize.value();
    resetRaggedForwardArrivalState();

    // Values are executed on Layer::stream. Join the structural producer before
    // the diagnostic kernel dereferences offsets[B].
    stream.waitFor(rowPartitionStream, rowPartitionReadyEvent);
    Layer::forward(featureInput, resolvedValidationPass, resolvedBatchSize);
}

void FiniteCheck::fuseBackwardAlias() {
    if (!errorInput.has_value() || !errorOutput.has_value())
        return;

    THOR_THROW_IF_FALSE(errorInput.value().getDescriptor() == errorOutput.value().getDescriptor());
    if (previousLayer.has_value())
        previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
    errorOutput = errorInput;
}

void FiniteCheck::validateRaggedInputs() const {
    THOR_THROW_IF_FALSE(raggedConfiguration.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(rowPartitionInput.has_value());
    const RaggedConfiguration& config = raggedConfiguration.value();
    if (featureInput->getTotalNumElements() != config.maxTotalValues * config.elementsPerValue) {
        throw std::runtime_error("Ragged FiniteCheck values input does not match its configured packed capacity.");
    }
    const TensorDescriptor offsetsDescriptor = rowPartitionInput->getDescriptor();
    if (offsetsDescriptor.getDimensions() != std::vector<uint64_t>{config.batchSize + 1} ||
        offsetsDescriptor.getDataType() != config.offsetsDataType ||
        !rowPartitionInput->isDenseContiguous() || rowPartitionInput->getStorageElementOffset() != 0) {
        throw std::runtime_error("Ragged FiniteCheck offsets input does not match its canonical row partition descriptor.");
    }
    if (featureInput->getPlacement() != rowPartitionInput->getPlacement()) {
        throw std::runtime_error("Ragged FiniteCheck values and offsets must have the same placement.");
    }
}

void FiniteCheck::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.has_value());
    if (raggedConfiguration.has_value()) validateRaggedInputs();

    if (!enabled)
        return;
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
    resetRaggedForwardArrivalState();
    rowPartitionReadyEvent = Event();
    Layer::cleanup();
}

void FiniteCheck::infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) {
    (void)outputTensor;
    if (!enabled)
        return;
    if (checkForward && inputTensor.has_value()) {
        if (raggedConfiguration.has_value())
            checkRaggedTensor(inputTensor.value(), "forward", "activation", stream);
        else
            checkTensor(inputTensor.value(), "forward", "activation", stream);
    }
}

void FiniteCheck::backProp(std::optional<Tensor> dataIn,
                           std::optional<Tensor> errorIn,
                           std::optional<Tensor> errorOut,
                           Stream stream) {
    (void)dataIn;
    (void)errorOut;
    if (!enabled)
        return;
    if (checkBackward && errorIn.has_value()) {
        if (raggedConfiguration.has_value())
            checkRaggedTensor(errorIn.value(), "backward", "incoming_gradient", stream);
        else
            checkTensor(errorIn.value(), "backward", "incoming_gradient", stream);
    }
}

void FiniteCheck::checkTensor(const Tensor &tensor, const char *direction, const char *tensorRole, Stream stream) {
    if (TensorDescriptor::isIntegralType(tensor.getDataType()) || tensor.getTotalNumElements() == 0)
        return;

    // Per-stamp host submission is serialized by contract. FiniteCheck is a
    // debugging barrier, so defensively serialize its private diagnostic workspace too.
    std::lock_guard<std::mutex> checkLock(checkMutex);
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
        result = checkCpuTensor(tensor, tensor.getTotalNumElements());
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

void FiniteCheck::checkRaggedTensor(const Tensor &tensor, const char *direction, const char *tensorRole, Stream stream) {
    THOR_THROW_IF_FALSE(raggedConfiguration.has_value());
    THOR_THROW_IF_FALSE(rowPartitionInput.has_value());
    if (TensorDescriptor::isIntegralType(tensor.getDataType()) || tensor.getTotalNumElements() == 0)
        return;

    std::lock_guard<std::mutex> checkLock(checkMutex);
    if (!tensor.isDenseContiguous()) {
        throw std::runtime_error("Ragged FiniteCheck requires dense contiguous packed values. label=\"" +
                                 (tensorLabel.empty() ? std::string("<unnamed>") : tensorLabel) + "\" direction=" + direction +
                                 " dtype=" + TensorDescriptor::getElementTypeName(tensor.getDataType()));
    }
    validateRaggedInputs();
    const RaggedConfiguration& config = raggedConfiguration.value();
    const uint64_t expectedElements = config.maxTotalValues * config.elementsPerValue;
    if (tensor.getTotalNumElements() != expectedElements ||
        tensor.getPlacement() != rowPartitionInput->getPlacement()) {
        throw std::runtime_error("Ragged FiniteCheck checked tensor does not match the configured packed values capacity/placement.");
    }

    FiniteCheckResult result{};
    if (tensor.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        stream.synchronize();
        RowPartitionRuntime rowPartition(
            rowPartitionInput.value(),
            RowPartitionDescriptor(config.batchSize, config.maxTotalValues, config.offsetsDataType));
        const uint64_t activeValues = rowPartition.requireHostActiveValueCount();
        const uint64_t activeElements = activeValues * config.elementsPerValue;
        result = checkCpuTensor(tensor, activeElements);
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
        launchRaggedFiniteCheck(tensor.getMemPtr(),
                                tensor.getDataType(),
                                rowPartitionInput->getMemPtr(),
                                config.offsetsDataType,
                                config.batchSize,
                                config.maxTotalValues,
                                config.elementsPerValue,
                                maxReportedIndices,
                                gpuResult,
                                stream);
        CUDA_CHECK(cudaMemcpyAsync(&result, gpuResult, sizeof(FiniteCheckResult), cudaMemcpyDeviceToHost, stream.getStream()));
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

FiniteCheckResult FiniteCheck::checkCpuTensor(const Tensor &tensor, uint64_t numElements) const {
    THOR_THROW_IF_FALSE(numElements <= tensor.getTotalNumElements());
    switch (tensor.getDataType()) {
        case DataType::FP8_E4M3:
            return checkCpuTyped(static_cast<const __nv_fp8_e4m3 *>(tensor.getMemPtr()), numElements, maxReportedIndices);
        case DataType::FP8_E5M2:
            return checkCpuTyped(static_cast<const __nv_fp8_e5m2 *>(tensor.getMemPtr()), numElements, maxReportedIndices);
        case DataType::FP16:
            return checkCpuTyped(static_cast<const half *>(tensor.getMemPtr()), numElements, maxReportedIndices);
        case DataType::BF16:
            return checkCpuTyped(static_cast<const __nv_bfloat16 *>(tensor.getMemPtr()), numElements, maxReportedIndices);
        case DataType::FP32:
            return checkCpuTyped(static_cast<const float *>(tensor.getMemPtr()), numElements, maxReportedIndices);
        case DataType::FP64:
            return checkCpuTyped(static_cast<const double *>(tensor.getMemPtr()), numElements, maxReportedIndices);
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
        << " checked_elements=" << result.checkedElements
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

void FiniteCheck::resetRaggedForwardArrivalState() {
    raggedValuesArrived = false;
    raggedPartitionArrived = false;
    pendingRaggedValidationPass.reset();
    pendingRaggedBatchSize.reset();
}

}  // namespace ThorImplementation
