#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {

bool isSupportedIndexType(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

bool isSupportedValueType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

}  // namespace

Embedding::Embedding(TensorPlacement placement,
                     std::vector<std::shared_ptr<PhysicalParameter>> parameters,
                     uint64_t vocabularySize,
                     uint64_t embeddingDim,
                     DataType weightsDataType,
                     std::optional<uint64_t> paddingIndex,
                     bool sparseGradients,
                     bool inferenceOnly,
                     int64_t stampedId)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      vocabularySize(vocabularySize),
      embeddingDim(embeddingDim),
      weightsDataType(weightsDataType),
      paddingIndex(paddingIndex),
      sparseGradients(sparseGradients) {
    if (vocabularySize == 0) {
        throw std::invalid_argument("Embedding vocabulary_size must be non-zero.");
    }
    if (embeddingDim == 0) {
        throw std::invalid_argument("Embedding embedding_dim must be non-zero.");
    }
    if (!isSupportedValueType(weightsDataType)) {
        throw std::invalid_argument("Embedding weights dtype must be fp16, bf16, or fp32. Got " + dtypeName(weightsDataType) + ".");
    }
    if (paddingIndex.has_value() && paddingIndex.value() >= vocabularySize) {
        throw std::invalid_argument("Embedding padding_index must be less than vocabulary_size.");
    }
    if (!sparseGradients) {
        throw std::invalid_argument("Embedding only supports sparse_gradients=true; dense gradients are intentionally not implemented.");
    }
    if (parameters.size() != 1 || parameters[0] == nullptr || parameters[0]->getName() != "weights") {
        throw std::invalid_argument("Embedding implementation requires exactly one parameter named 'weights'.");
    }
    this->parameters = std::move(parameters);
    parameterIndexByName.clear();
    parameterIndexByName["weights"] = 0;
}

void Embedding::compileImpl() {
    TrainableLayer::compileImpl();

    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Embedding currently requires GPU placement.");
    }

    std::optional<Tensor> aFeatureInput = getFirstPresentTensor(featureInputs);
    if (!aFeatureInput.has_value()) {
        throw std::invalid_argument("Embedding requires at least one connected feature input.");
    }
    if (!isSupportedIndexType(aFeatureInput.value().getDataType())) {
        throw std::invalid_argument("Embedding indices dtype must be uint8, uint16, uint32, or uint64. Got " +
                                    dtypeName(aFeatureInput.value().getDataType()) + ".");
    }

    PhysicalParameter::StorageContext storageContext = buildParameterStorageContext();
    for (const auto& parameter : parameters) {
        if (!parameter->isStorageInitialized()) {
            parameter->compileStorage(storageContext);
        }
        THOR_THROW_IF_FALSE(parameter->getStorage().has_value());
        Tensor storage = parameter->getStorage().value();
        if (storage.getDimensions() != std::vector<uint64_t>{vocabularySize, embeddingDim}) {
            throw std::invalid_argument("Embedding weights storage shape does not match [vocabulary_size, embedding_dim].");
        }
        if (storage.getDataType() != weightsDataType) {
            throw std::invalid_argument("Embedding weights storage dtype does not match weightsDataType.");
        }
        parameter->compileInitializer(1, embeddingDim);
    }

    attachGradientUpdateStream();

    initializeEmbeddingKernelsSharedAttributes();

    // Do not call PhysicalParameter::compileOptimizer here. The default optimizer path materializes a dense
    // weightsGradient tensor, which is exactly what Embedding must avoid for large vocabularies. Embedding instead
    // asks the optimizer for an optimizer-owned reduced sparse-gradient sink and sparse-row update plan.
    if (gradientUpdateStream.has_value()) {
        for (const auto& parameter : parameters) {
            if (!isInferenceOnly() && parameter->isTrainingEnabled()) {
                if (!parameter->hasOptimizer()) {
                    throw std::invalid_argument("Embedding trainable weights require an optimizer.");
                }
                std::shared_ptr<Optimizer> optimizer = parameter->getOptimizer();
                if (!optimizer->supportsSparseRowGradients()) {
                    throw std::invalid_argument(
                        "Embedding weights produce reduced sparse row gradients, but the attached optimizer does not support sparse row "
                        "gradients. Dense-gradient fallback is intentionally forbidden.");
                }

                if (numBackwardConnections != 1) {
                    throw std::invalid_argument(
                        "Embedding sparse-gradient reduction currently supports exactly one backward connection. Multiple backward "
                        "connections "
                        "require merging reduced SparseRowGradient sinks before a single optimizer-state update, and silent per-connection "
                        "updates are intentionally forbidden.");
                }
                std::optional<Tensor> aErrorInput = getFirstPresentTensor(errorInputs);
                if (!aErrorInput.has_value()) {
                    throw std::invalid_argument("Trainable Embedding requires an error input so it can produce sparse row gradients.");
                }

                const Tensor storage = parameter->getStorage().value();
                const uint64_t maxSparseRows = std::min<uint64_t>(aFeatureInput.value().getTotalNumElements(), vocabularySize);
                if (!optimizer->supportsSparseRowUpdateFusion()) {
                    throw std::invalid_argument(
                        "Embedding production training requires an optimizer with fused sparse-row update support. The legacy "
                        "materialized SparseRowGradient update path has been removed from Embedding.");
                }
                if (!supportsEmbeddingSparseGradientFusedSparseRowUpdate(embeddingDim)) {
                    throw std::invalid_argument(
                        "Embedding production training requires fused sparse-row update support for embedding_dim=" +
                        std::to_string(embeddingDim) + ". The legacy materialized SparseRowGradient update path has been removed from "
                        "Embedding.");
                }

                weightsSparseGradient = optimizer->compileSparseRows(storage, maxSparseRows, gradientUpdateStream.value());
                THOR_THROW_IF_FALSE(weightsSparseGradient.has_value());

                SparseRowOptimizerExpression updateExpression =
                    optimizer->toSparseRowUpdateExpression(storage, weightsSparseGradient.value());
                weightsSparseGradientProducer = prepareEmbeddingSparseGradientWithSparseRowUpdate(aFeatureInput.value(),
                                                                                                  aErrorInput.value(),
                                                                                                  weightsSparseGradient.value(),
                                                                                                  updateExpression.outputs,
                                                                                                  updateExpression.inputs,
                                                                                                  updateExpression.indexedOutputs,
                                                                                                  paddingIndex);

                weightsSparseGradientCapturedGraph.emplace(placement.getDeviceNum());
                CudaGraphCaptureBuilder builder(gradientUpdateStream.value());
                capturePreparedEmbeddingSparseGradientWithSparseRowUpdateRuntimeScalarStorage(builder,
                                                                                              *weightsSparseGradientProducer,
                                                                                              aFeatureInput.value(),
                                                                                              aErrorInput.value(),
                                                                                              weightsSparseGradient.value(),
                                                                                              weightsSparseGradientCapturedGraph.value());
                weightsSparseGradientGraphExecutable.emplace(
                    endCaptureAndInstantiatePreparedEmbeddingSparseGradientGraph(builder,
                                                                                 weightsSparseGradientCapturedGraph.value(),
                                                                                 gradientUpdateStream.value()));
                weightsSparseGradientCapturedGraph->uploadTargetNodes(gradientUpdateStream.value());
            }
        }
    }
}

std::optional<Tensor> Embedding::createFeatureOutputTensor() {
    std::optional<Tensor> featureInput = getFirstPresentTensor(featureInputs);
    THOR_THROW_IF_FALSE(featureInput.has_value());

    std::vector<uint64_t> outputDims = featureInput.value().getDimensions();
    outputDims.push_back(embeddingDim);
    return Tensor(featureInput.value().getPlacement(), TensorDescriptor(weightsDataType, outputDims));
}

std::optional<Tensor> Embedding::createErrorOutputTensor(bool backPropagateError, uint32_t connectionNumber) {
    (void)backPropagateError;
    (void)connectionNumber;
    // Indices are discrete; there is no meaningful gradient to propagate to the previous layer.
    return std::nullopt;
}

Tensor Embedding::weights() const {
    THOR_THROW_IF_FALSE(parameters.size() == 1);
    std::optional<Tensor> storage = parameters[0]->getStorage();
    THOR_THROW_IF_FALSE(storage.has_value());
    return storage.value();
}

void Embedding::computeFeatureOut(uint32_t connectionNumber) {
    THOR_THROW_IF_FALSE(connectionNumber < featureInputs.size());
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());
    THOR_THROW_IF_FALSE(featureOutputs[connectionNumber].has_value());

    Tensor weightsTensor = weights();
    launchEmbeddingForward(featureInputs[connectionNumber].value(),
                           weightsTensor,
                           featureOutputs[connectionNumber].value(),
                           paddingIndex,
                           streams[connectionNumber]);
}

void Embedding::backward(std::optional<Tensor> errorInput, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);
    if (!errorInput.has_value()) {
        return;
    }

    uint32_t connectionNumber = 0;
    for (; connectionNumber < errorInputs.size(); ++connectionNumber) {
        if (errorInputs[connectionNumber].has_value() && errorInput.value() == errorInputs[connectionNumber].value())
            break;
    }
    THOR_THROW_IF_FALSE(connectionNumber != errorInputs.size());
    THOR_THROW_IF_FALSE(featureInputs[connectionNumber].has_value());

    if (isStartOfBackward) {
        isStartOfBackward = false;
    }

    if (!isInferenceOnly() && gradientUpdateStream.has_value() && parameters[0]->isTrainingEnabled()) {
        Event errorInputReady = streams[connectionNumber].putEvent();
        gradientUpdateStream.value().waitEvent(errorInputReady);
        THOR_THROW_IF_FALSE(weightsSparseGradient.has_value());
        THOR_THROW_IF_FALSE(weightsSparseGradientProducer != nullptr);
        THOR_THROW_IF_FALSE(weightsSparseGradientGraphExecutable.has_value());
        THOR_THROW_IF_FALSE(weightsSparseGradientCapturedGraph.has_value());
        updateCapturedEmbeddingSparseGradientSparseRowUpdateRuntimeScalars(
            *weightsSparseGradientProducer,
            weightsSparseGradientCapturedGraph.value(),
            weightsSparseGradientGraphExecutable.value(),
            parameters[0]->getOptimizer()->sparseRowUpdateRuntimeScalars(batchSize * numBackwardConnections));
        weightsSparseGradientGraphExecutable->launch(gradientUpdateStream.value());
    }

    numBackwardConnectionsMade += 1;
    bool gradientComplete = false;
    if (numBackwardConnectionsMade == numBackwardConnections) {
        gradientComplete = true;
        numBackwardConnectionsMade = 0;
    }
    THOR_THROW_IF_FALSE(numBackwardConnectionsMade < numBackwardConnections);

    if (gradientComplete) {
        weightsAreUpToDateEvent.reset();
        if (!isInferenceOnly() && gradientUpdateStream.has_value() && parameters[0]->isTrainingEnabled()) {
            weightsAreUpToDateEvent = gradientUpdateStream.value().putEvent();
        }
        isStartOfForward = true;
    }

    // No previous-layer backward call: Embedding has no gradient with respect to integer indices.
}

}  // namespace ThorImplementation
