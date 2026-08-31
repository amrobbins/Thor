#include "DeepLearning/Implementation/Layers/NeuralNetwork/Embedding.h"
#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

#include <algorithm>
#include <limits>
#include <set>
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
                     int64_t stampedId,
                     std::optional<RaggedEmbeddingConfig> raggedConfig)
    : TrainableLayer(placement, inferenceOnly, stampedId),
      vocabularySize(vocabularySize),
      embeddingDim(embeddingDim),
      weightsDataType(weightsDataType),
      paddingIndex(paddingIndex),
      sparseGradients(sparseGradients),
      raggedConfig(std::move(raggedConfig)) {
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
    if (this->raggedConfig.has_value()) {
        const RaggedEmbeddingConfig& config = this->raggedConfig.value();
        if (config.batchSize == 0 || config.maxTotalValues == 0 || config.elementsPerValue == 0) {
            throw std::invalid_argument("Ragged Embedding requires non-zero batch/capacity/elements-per-value metadata.");
        }
        if (config.offsetsDataType != DataType::UINT32 && config.offsetsDataType != DataType::UINT64) {
            throw std::invalid_argument("Ragged Embedding offsets dtype must be uint32 or uint64.");
        }
    }
    if (parameters.size() != 1 || parameters[0] == nullptr || parameters[0]->getName() != "weights") {
        throw std::invalid_argument("Embedding implementation requires exactly one parameter named 'weights'.");
    }
    this->parameters = std::move(parameters);
    parameterIndexByName.clear();
    parameterIndexByName["weights"] = 0;
}

uint32_t Embedding::raggedApplicationCount() const {
    if (!isRagged())
        return 0;
    return static_cast<uint32_t>(std::max(featureOutputs.size(), featureInputs.size() / 2));
}

void Embedding::ensureRaggedApplicationStorage(uint32_t applicationIndex) {
    const size_t requiredInputs = static_cast<size_t>(applicationIndex + 1) * 2;
    const size_t requiredOutputs = static_cast<size_t>(applicationIndex + 1);
    if (featureInputs.size() < requiredInputs) featureInputs.resize(requiredInputs, std::nullopt);
    if (errorOutputs.size() < requiredInputs) errorOutputs.resize(requiredInputs, std::nullopt);
    if (previousLayers.size() < requiredInputs) previousLayers.resize(requiredInputs, std::nullopt);
    if (streams.size() < requiredInputs) streams.resize(requiredInputs);
    if (featureOutputs.size() < requiredOutputs) featureOutputs.resize(requiredOutputs, std::nullopt);
    if (errorInputs.size() < requiredOutputs) errorInputs.resize(requiredOutputs, std::nullopt);
    if (nextLayers.size() < requiredOutputs) nextLayers.resize(requiredOutputs, std::nullopt);
    if (raggedRuntimeExtents.size() < requiredOutputs) raggedRuntimeExtents.resize(requiredOutputs);
    if (preparedRaggedForwards.size() < requiredOutputs) preparedRaggedForwards.resize(requiredOutputs);
    if (raggedAllForwardInputTensorIds.size() < requiredOutputs) raggedAllForwardInputTensorIds.resize(requiredOutputs);
    if (raggedWaitingForwardInputTensorIds.size() < requiredOutputs) raggedWaitingForwardInputTensorIds.resize(requiredOutputs);
    if (raggedCurrentValidExampleCounts.size() < requiredOutputs) raggedCurrentValidExampleCounts.resize(requiredOutputs, 0);
    if (raggedBatchCardinalitySet.size() < requiredOutputs) raggedBatchCardinalitySet.resize(requiredOutputs, false);
    if (raggedOffsetsReadyEvents.size() < requiredOutputs) raggedOffsetsReadyEvents.resize(requiredOutputs);
}

void Embedding::resetRaggedForwardArrivalBookkeeping() {
    if (!isRagged())
        return;
    for (uint32_t app = 0; app < raggedApplicationCount(); ++app) {
        ensureRaggedApplicationStorage(app);
        raggedAllForwardInputTensorIds[app].clear();
        const uint32_t valuesSlot = raggedValuesSlot(app);
        const uint32_t offsetsSlot = raggedOffsetsSlot(app);
        if (valuesSlot < featureInputs.size() && featureInputs[valuesSlot].has_value())
            raggedAllForwardInputTensorIds[app].insert(featureInputs[valuesSlot]->getTensorId());
        if (offsetsSlot < featureInputs.size() && featureInputs[offsetsSlot].has_value())
            raggedAllForwardInputTensorIds[app].insert(featureInputs[offsetsSlot]->getTensorId());
        raggedWaitingForwardInputTensorIds[app] = raggedAllForwardInputTensorIds[app];
        raggedCurrentValidExampleCounts[app] = 0;
        raggedBatchCardinalitySet[app] = false;
    }
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

    if (isRagged()) {
        const RaggedEmbeddingConfig& config = raggedConfig.value();
        const uint32_t applications = raggedApplicationCount();
        if (applications == 0) {
            throw std::invalid_argument("Ragged Embedding requires at least one connected application.");
        }
        Tensor weightsTensor = weights();
        for (uint32_t app = 0; app < applications; ++app) {
            ensureRaggedApplicationStorage(app);
            const uint32_t valuesSlot = raggedValuesSlot(app);
            const uint32_t offsetsSlot = raggedOffsetsSlot(app);
            if (!featureInputs[valuesSlot].has_value() || !featureInputs[offsetsSlot].has_value() ||
                !featureOutputs[app].has_value()) {
                throw std::invalid_argument("Ragged Embedding requires values, offsets, and output for every application.");
            }
            const Tensor& indices = featureInputs[valuesSlot].value();
            const Tensor& offsets = featureInputs[offsetsSlot].value();
            if (indices.getDimensions().empty() || indices.getDimensions()[0] != config.maxTotalValues) {
                throw std::invalid_argument("Ragged Embedding indices packed capacity does not match max_total_values.");
            }
            if (offsets.getDimensions() != std::vector<uint64_t>{config.batchSize + 1} ||
                offsets.getDataType() != config.offsetsDataType) {
                throw std::invalid_argument("Ragged Embedding offsets descriptor does not match its row-partition metadata.");
            }
            if (config.maxTotalValues > std::numeric_limits<uint64_t>::max() / config.elementsPerValue ||
                indices.getTotalNumElements() != config.maxTotalValues * config.elementsPerValue) {
                throw std::invalid_argument("Ragged Embedding indices trailing geometry does not match elements_per_value.");
            }
            raggedRuntimeExtents[app] =
                raggedRuntimeExtentFromOffsets(offsets, config.batchSize, config.maxTotalValues, config.elementsPerValue);
            preparedRaggedForwards[app] = prepareEmbeddingForwardRagged(indices,
                                                                        weightsTensor,
                                                                        featureOutputs[app].value(),
                                                                        paddingIndex,
                                                                        raggedRuntimeExtents[app]);
        }
        resetRaggedForwardArrivalBookkeeping();
    }

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

                uint32_t trainingApplication = 0;
                if (isRagged()) {
                    for (; trainingApplication < errorInputs.size(); ++trainingApplication) {
                        if (errorInputs[trainingApplication].has_value()) break;
                    }
                    THOR_THROW_IF_FALSE(trainingApplication < errorInputs.size());
                    aFeatureInput = featureInputs[raggedValuesSlot(trainingApplication)];
                    THOR_THROW_IF_FALSE(aFeatureInput.has_value());
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
                if (isRagged()) {
                    weightsSparseGradientProducer = prepareEmbeddingSparseGradientWithSparseRowUpdateRagged(
                        aFeatureInput.value(),
                        aErrorInput.value(),
                        weightsSparseGradient.value(),
                        updateExpression.outputs,
                        updateExpression.inputs,
                        updateExpression.indexedOutputs,
                        paddingIndex,
                        raggedRuntimeExtents[trainingApplication]);
                } else {
                    weightsSparseGradientProducer = prepareEmbeddingSparseGradientWithSparseRowUpdate(aFeatureInput.value(),
                                                                                                      aErrorInput.value(),
                                                                                                      weightsSparseGradient.value(),
                                                                                                      updateExpression.outputs,
                                                                                                      updateExpression.inputs,
                                                                                                      updateExpression.indexedOutputs,
                                                                                                      paddingIndex);
                }

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

void Embedding::initialize() {
    TrainableLayer::initialize();
    if (isRagged()) resetRaggedForwardArrivalBookkeeping();
}

void Embedding::cleanup() {
    preparedRaggedForwards.clear();
    raggedRuntimeExtents.clear();
    raggedAllForwardInputTensorIds.clear();
    raggedWaitingForwardInputTensorIds.clear();
    raggedCurrentValidExampleCounts.clear();
    raggedBatchCardinalitySet.clear();
    raggedOffsetsReadyEvents.clear();
    weightsSparseGradientProducer.reset();
    weightsSparseGradient.reset();
    weightsSparseGradientCapturedGraph.reset();
    weightsSparseGradientGraphExecutable.reset();
    TrainableLayer::cleanup();
}

std::optional<Tensor> Embedding::connectToPreviousLayer(
    Layer* previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType) {
    if (!isRagged()) {
        return TrainableLayer::connectToPreviousLayer(previousLayer, featureInput, stream, backPropagateError, connectionType);
    }

    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(previousLayer != nullptr && featureInput.has_value());
    if (connectionType < 0) throw std::invalid_argument("Ragged Embedding input connection type must be non-negative.");
    const uint32_t encoded = static_cast<uint32_t>(connectionType);
    const uint32_t applicationIndex = encoded / 2;
    const uint32_t portIndex = encoded % 2;
    ensureRaggedApplicationStorage(applicationIndex);
    const uint32_t flat = portIndex == 0 ? raggedValuesSlot(applicationIndex) : raggedOffsetsSlot(applicationIndex);
    if (featureInputs[flat].has_value() || previousLayers[flat].has_value()) {
        throw std::invalid_argument("Ragged Embedding input port was connected more than once.");
    }

    const RaggedEmbeddingConfig& config = raggedConfig.value();
    if (portIndex == 0) {
        if (!isSupportedIndexType(featureInput->getDataType())) {
            throw std::invalid_argument("Ragged Embedding indices dtype must be uint8, uint16, uint32, or uint64.");
        }
        const std::vector<uint64_t>& dims = featureInput->getDimensions();
        if (dims.empty() || dims[0] != config.maxTotalValues) {
            throw std::invalid_argument("Ragged Embedding values input must use the configured packed capacity.");
        }
    } else {
        if (featureInput->getDimensions() != std::vector<uint64_t>{config.batchSize + 1} ||
            featureInput->getDataType() != config.offsetsDataType) {
            throw std::invalid_argument("Ragged Embedding offsets input does not match the configured row partition.");
        }
    }

    previousLayers[flat] = previousLayer;
    featureInputs[flat] = featureInput;
    streams[flat] = stream;
    errorOutputs[flat] = std::nullopt;  // integer indices and structural offsets have no gradient
    ensureNoDeviceCrossing(placement);
    (void)backPropagateError;
    return std::nullopt;
}

void Embedding::connectToNextLayer(Layer* nextLayer, int driverConnectionType, int loaderConnectionType) {
    if (!isRagged()) {
        TrainableLayer::connectToNextLayer(nextLayer, driverConnectionType, loaderConnectionType);
        return;
    }

    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(nextLayer != nullptr);
    if (driverConnectionType < 0) throw std::invalid_argument("Ragged Embedding output connection type must be non-negative.");
    const uint32_t applicationIndex = static_cast<uint32_t>(driverConnectionType);
    ensureRaggedApplicationStorage(applicationIndex);
    const uint32_t valuesSlot = raggedValuesSlot(applicationIndex);
    const uint32_t offsetsSlot = raggedOffsetsSlot(applicationIndex);
    if (!featureInputs[valuesSlot].has_value() || !featureInputs[offsetsSlot].has_value()) {
        throw std::invalid_argument("Ragged Embedding output cannot be connected until both values and offsets inputs are connected.");
    }

    if (!featureOutputs[applicationIndex].has_value()) {
        std::vector<uint64_t> outputDims = featureInputs[valuesSlot]->getDimensions();
        outputDims.push_back(embeddingDim);
        featureOutputs[applicationIndex] =
            Tensor(featureInputs[valuesSlot]->getPlacement(), TensorDescriptor(weightsDataType, outputDims));
    }
    nextLayers[applicationIndex] = nextLayer;
    errorInputs[applicationIndex] = nextLayer->connectToPreviousLayer(this,
                                                                      featureOutputs[applicationIndex],
                                                                      streams[valuesSlot],
                                                                      shouldConnectToBackPropErrorIn(),
                                                                      loaderConnectionType);
    ensureNoDeviceCrossing(placement);
}

void Embedding::forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t runtimeBatchSize) {
    if (!isRagged()) {
        TrainableLayer::forward(featureInput, validationPass, runtimeBatchSize);
        return;
    }

    THOR_THROW_IF_FALSE(running && featureInput.has_value());
    const RaggedEmbeddingConfig& config = raggedConfig.value();
    THOR_THROW_IF_FALSE(config.batchSize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
    const uint32_t physicalBatchCapacity = static_cast<uint32_t>(config.batchSize);
    const uint32_t resolvedValidExampleCount = runtimeBatchSize == 0 ? physicalBatchCapacity : runtimeBatchSize;
    THOR_THROW_IF_FALSE(resolvedValidExampleCount >= 1 && resolvedValidExampleCount <= physicalBatchCapacity);

    if (isStartOfForward) {
        if (weightsAreUpToDateEventValid) {
            for (const Stream& dataStream : uniqueDataStreams) dataStream.waitEvent(weightsAreUpToDateEvent);
        }
        weightsAreUpToDateEventValid = false;
        isStartOfForward = false;
        isStartOfBackward = true;
        resetRaggedForwardArrivalBookkeeping();
    }

    std::set<uint32_t> candidateApplications;
    for (uint32_t app = 0; app < raggedApplicationCount(); ++app) {
        const uint32_t valuesSlot = raggedValuesSlot(app);
        const uint32_t offsetsSlot = raggedOffsetsSlot(app);
        if ((featureInputs[valuesSlot].has_value() && featureInputs[valuesSlot].value() == featureInput.value()) ||
            (featureInputs[offsetsSlot].has_value() && featureInputs[offsetsSlot].value() == featureInput.value())) {
            candidateApplications.insert(app);
        }
    }
    THOR_THROW_IF_FALSE(!candidateApplications.empty());

    for (uint32_t app : candidateApplications) {
        if (raggedBatchCardinalitySet[app]) {
            THOR_THROW_IF_FALSE(raggedCurrentValidExampleCounts[app] == resolvedValidExampleCount);
        } else {
            raggedCurrentValidExampleCounts[app] = resolvedValidExampleCount;
            raggedBatchCardinalitySet[app] = true;
        }

        const uint64_t tensorId = featureInput->getTensorId();
        auto waitingIt = raggedWaitingForwardInputTensorIds[app].find(tensorId);
        if (waitingIt == raggedWaitingForwardInputTensorIds[app].end()) continue;
        raggedWaitingForwardInputTensorIds[app].erase(waitingIt);
        if (!raggedWaitingForwardInputTensorIds[app].empty()) continue;

        const uint32_t valuesSlot = raggedValuesSlot(app);
        const uint32_t offsetsSlot = raggedOffsetsSlot(app);
        streams[valuesSlot].waitFor(streams[offsetsSlot], raggedOffsetsReadyEvents[app]);
        computeFeatureOut(app);
        if (nextLayers[app].has_value()) {
            nextLayers[app].value()->forward(featureOutputs[app], validationPass, raggedCurrentValidExampleCounts[app]);
        }

        raggedWaitingForwardInputTensorIds[app] = raggedAllForwardInputTensorIds[app];
        raggedCurrentValidExampleCounts[app] = 0;
        raggedBatchCardinalitySet[app] = false;
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
    if (isRagged()) {
        const uint32_t applicationIndex = connectionNumber;
        const uint32_t valuesSlot = raggedValuesSlot(applicationIndex);
        THOR_THROW_IF_FALSE(applicationIndex < preparedRaggedForwards.size());
        THOR_THROW_IF_FALSE(preparedRaggedForwards[applicationIndex] != nullptr);
        THOR_THROW_IF_FALSE(valuesSlot < featureInputs.size() && featureInputs[valuesSlot].has_value());
        THOR_THROW_IF_FALSE(applicationIndex < featureOutputs.size() && featureOutputs[applicationIndex].has_value());
        Tensor weightsTensor = weights();
        launchPreparedEmbeddingForward(*preparedRaggedForwards[applicationIndex],
                                       featureInputs[valuesSlot].value(),
                                       weightsTensor,
                                       featureOutputs[applicationIndex].value(),
                                       streams[valuesSlot]);
        return;
    }

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
    const uint32_t valuesSlot = isRagged() ? raggedValuesSlot(connectionNumber) : connectionNumber;
    THOR_THROW_IF_FALSE(valuesSlot < featureInputs.size() && featureInputs[valuesSlot].has_value());

    if (isStartOfBackward) {
        isStartOfBackward = false;
    }

    if (!isInferenceOnly() && gradientUpdateStream.has_value() && parameters[0]->isTrainingEnabled()) {
        THOR_THROW_IF_FALSE(valuesSlot < errorInputReadyEvents.size());
        streams[valuesSlot].putEvent(errorInputReadyEvents[valuesSlot]);
        gradientUpdateStream.value().waitEvent(errorInputReadyEvents[valuesSlot]);
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
        weightsAreUpToDateEventValid = false;
        if (!isInferenceOnly() && gradientUpdateStream.has_value() && parameters[0]->isTrainingEnabled()) {
            gradientUpdateStream.value().putEvent(weightsAreUpToDateEvent);
            weightsAreUpToDateEventValid = true;
        }
        isStartOfForward = true;
    }

    // No previous-layer backward call: Embedding has no gradient with respect to integer indices.
}

}  // namespace ThorImplementation
