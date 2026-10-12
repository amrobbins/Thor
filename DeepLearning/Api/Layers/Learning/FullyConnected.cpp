#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"
#include "Utilities/Expression/DropOutPostOp.h"
#include "Utilities/Expression/ExpressionInternal.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <cmath>
#include <random>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <optional>
#include <set>
#include <sstream>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

constexpr const char* RAGGED_ROW_PARTITION_EXPRESSION_INPUT = "__ragged_row_partition_runtime";
constexpr const char* FC_RESIDUAL_INPUT_NAME = "residual_input";
constexpr const char* FC_OUTPUT_DROPOUT_SEED_INPUT_NAME = "__fully_connected_output_dropout_seed";
constexpr const char* FC_OUTPUT_DROPOUT_SEQUENCE_INPUT_NAME = "__fully_connected_output_dropout_sequence";
constexpr ThorImplementation::DynamicExpressionVariantId FC_EVALUATION_VARIANT = 1;

bool supportedRaggedFcStorageType(DataType dataType) {
    return dataType == DataType::FP16 || dataType == DataType::BF16 || dataType == DataType::FP32;
}

// Version 1.3.0 is the basic logical ragged projection, including multiple applications.
// Version 1.4.0 adds an activation template, including multiple applications.
// Version 1.5.0 adds first-class residual with an identical canonical partition;
// its existing activation and dropout fields can compose with the residual.
// An application count of one retains the two historical residual reference
// objects. Multiple applications carry ordered arrays of the same reference
// grammar under those existing keys, validated per independent partition.
// Version 1.6.0 adds output dropout (optionally following activation), including multiple applications.
// Version 1.7.0 adds a custom epilogue with any number of same-partition
// auxiliary inputs; its existing activation, dropout, and residual fields may
// compose. Multiple applications repeat the existing named binding objects in
// application-major order, one complete set per independent partition.
// These are extensions of existing optional fields, not new grammars.
// Older 1.0/1.1/1.2 are historical input grammars; they do not select a geometry mode.
void validateNeutralFcTensorReference(const json& record, const Tensor& live) {
    if (!record.is_object() || record.size() != 4 ||
        record.at("version").get<std::string>() != "1.0.0" ||
        record.at("id").get<uint64_t>() != live.getOriginalId() ||
        record.at("data_type").get<DataType>() != live.getDataType() ||
        record.at("dimensions").get<std::vector<uint64_t>>() != live.getDimensions()) {
        throw std::runtime_error("FullyConnected neutral tensor reference differs from its live graph tensor.");
    }
}

void validateNeutralFcRaggedReference(const json& record, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(record) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(record, "FullyConnected") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        record.at("ragged_rank").get<uint32_t>() != 1 ||
        record.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        record.contains("batch_size") || record.contains("max_total_values")) {
        throw std::runtime_error("FullyConnected neutral ragged reference has invalid semantic geometry.");
    }
    validateNeutralFcTensorReference(record.at("values"), live.getValues());
    validateNeutralFcTensorReference(record.at("offsets"), live.getRowPartitionToken());
}

bool isFullyConnectedFloatingDataType(DataType dataType) {
    switch (dataType) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

bool isFullyConnectedFp8DataType(DataType dataType) {
    return dataType == DataType::FP8_E4M3 || dataType == DataType::FP8_E5M2;
}

std::string fullyConnectedDataTypeName(DataType dataType) {
    return ThorImplementation::TensorDescriptor::getElementTypeName(dataType);
}

cudaDataType_t cublasLtCudaDataTypeForFullyConnected(DataType dataType) {
    switch (dataType) {
        case DataType::FP32:
            return CUDA_R_32F;
        case DataType::BF16:
            return CUDA_R_16BF;
        case DataType::FP16:
            return CUDA_R_16F;
        case DataType::FP8_E4M3:
            return CUDA_R_8F_E4M3;
        case DataType::FP8_E5M2:
            return CUDA_R_8F_E5M2;
        default:
            throw std::invalid_argument("FullyConnected cuBLASLt dtype check does not support " +
                                        fullyConnectedDataTypeName(dataType) + ".");
    }
}

std::optional<cublasComputeType_t> cublasLtComputeTypeForFullyConnected(DataType computeDataType) {
    switch (computeDataType) {
        case DataType::FP32:
            return CUBLAS_COMPUTE_32F;
        case DataType::TF32:
            return CUBLAS_COMPUTE_32F_FAST_TF32;
        case DataType::FP16:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        case DataType::BF16:
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        default:
            return std::nullopt;
    }
}

bool isSupportedCublasLtMatmulDataTypesForFullyConnected(
    const ThorImplementation::CublasMatrixMultiply::MatmulDataTypes& dataTypes) {
    if (!isFullyConnectedFloatingDataType(dataTypes.A) ||
        !isFullyConnectedFloatingDataType(dataTypes.B) ||
        !isFullyConnectedFloatingDataType(dataTypes.C) ||
        !isFullyConnectedFloatingDataType(dataTypes.D)) {
        return false;
    }

    const std::optional<cublasComputeType_t> computeType = cublasLtComputeTypeForFullyConnected(dataTypes.compute);
    if (!computeType.has_value()) {
        return false;
    }

    const cudaDataType_t ADataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.A);
    const cudaDataType_t BDataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.B);
    const cudaDataType_t CDataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.C);
    const cudaDataType_t DDataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.D);

    return isSupportedCublasLtOperationType(computeType.value(), CUDA_R_32F, ADataType, BDataType, CDataType, DDataType);
}

ThorImplementation::CublasMatrixMultiply::MatmulDataTypes cublasLtMatmulDataTypesForFullyConnected(
    DataType inputDataType,
    DataType weightsDataType,
    DataType computeDataType,
    DataType outputDataType) {
    using MatmulDataTypes = ThorImplementation::CublasMatrixMultiply::MatmulDataTypes;

    const bool usesFp8Storage = isFullyConnectedFp8DataType(inputDataType) || isFullyConnectedFp8DataType(weightsDataType) ||
                                isFullyConnectedFp8DataType(outputDataType);
    if (usesFp8Storage && computeDataType != DataType::FP32) {
        throw std::invalid_argument(
            "FullyConnected FP8 storage requires computeDataType fp32 for Thor's cuBLASLt GEMM path. Got " +
            fullyConnectedDataTypeName(computeDataType) + ".");
    }

    const MatmulDataTypes directDataTypes{inputDataType, weightsDataType, outputDataType, outputDataType, computeDataType};
    if (isSupportedCublasLtMatmulDataTypesForFullyConnected(directDataTypes)) {
        return directDataTypes;
    }

    throw std::invalid_argument(
        "FullyConnected requested dtype plan is unsupported by Thor's cuBLASLt matmul path and Thor will not implicitly "
        "convert either operand. Add an explicit network conversion or choose a directly supported plan. input=" +
        fullyConnectedDataTypeName(inputDataType) + ", weights=" + fullyConnectedDataTypeName(weightsDataType) +
        ", compute=" + fullyConnectedDataTypeName(computeDataType) + ", output=" +
        fullyConnectedDataTypeName(outputDataType) + ".");
}

std::string dimensionsString(const std::vector<uint64_t>& dimensions) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0)
            out << ", ";
        out << dimensions[i];
    }
    out << "]";
    return out.str();
}

using ExpressionInputDataTypes = ThorImplementation::ExpressionInputAnnotations;

ExpressionInputDataTypes expressionInputDataTypes(const ThorImplementation::Expression& expression,
                                                   const std::string& inputName) {
    const std::vector<ExpressionInputDataTypes> annotations =
        ThorImplementation::ExpressionInternalAccess::inputAnnotations(expression, inputName);
    if (annotations.empty()) {
        throw std::runtime_error("FullyConnected epilogue expression does not contain expected input '" + inputName + "'.");
    }

    const ExpressionInputDataTypes& resolved = annotations.front();
    for (const ExpressionInputDataTypes& candidate : annotations) {
        if (resolved != candidate) {
            throw std::runtime_error("FullyConnected epilogue input '" + inputName +
                                     "' is used with inconsistent dtype annotations.");
        }
    }
    return resolved;
}

ThorImplementation::DynamicExpression buildFullyConnectedExpression(uint64_t apiLayerId,
                                                                    bool hasBias,
                                                                    bool preserveInputPrefixDimensions,
                                                                    ThorImplementation::TensorPlacement placement,
                                                                    DataType weightsDataType,
                                                                    DataType computeDataType,
                                                                    DataType outputDataType,
                                                                    std::shared_ptr<Thor::Activation> activation,
                                                                    std::optional<uint64_t> packedRowCapacity,
                                                                    std::vector<uint64_t> applicationMaxValuesPerRow,
                                                                    std::optional<std::string> rowPartitionInputName,
                                                                    std::optional<ThorImplementation::RowPartitionDescriptor> rowPartitionDescriptor,
                                                                    std::optional<ThorImplementation::Expression> epilogue,
                                                                    std::vector<std::string> epilogueAuxInputNames,
                                                                    float outputDropoutProbability,
                                                                    int64_t outputDropoutSeed,
                                                                    bool useResidual) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::DynamicExpressionVariant;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;
    using ThorImplementation::TensorScalarBinding;

    std::vector<std::string> expectedInputNames = {"feature_input"};
    if (rowPartitionInputName.has_value()) {
        expectedInputNames.push_back(rowPartitionInputName.value());
    } else {
        // Dense FullyConnected executes against physical batch-capacity tensors,
        // but exact partial batches make only the leading valid-example prefix
        // semantically live.  Consume CustomLayer's runtime validity mask so
        // inactive source storage can remain genuinely unspecified.
        expectedInputNames.push_back(Thor::BATCH_VALIDITY_MASK_NAME);
    }
    if (useResidual) {
        expectedInputNames.push_back(FC_RESIDUAL_INPUT_NAME);
    }
    expectedInputNames.insert(expectedInputNames.end(), epilogueAuxInputNames.begin(), epilogueAuxInputNames.end());
    expectedInputNames.push_back("weights");
    if (hasBias) {
        expectedInputNames.push_back("biases");
    }

    std::shared_ptr<Thor::Activation> activationClone = nullptr;
    if (activation != nullptr) {
        activationClone = std::dynamic_pointer_cast<Thor::Activation>(activation->clone());
        if (activationClone == nullptr) {
            throw std::runtime_error("FullyConnected activation clone did not produce an Activation.");
        }
    }

    // Output discovery has no output tensor yet, whereas final stamping has a
    // unique output per physical application. Allocate independent dropout
    // substreams by that output identity, never by feature-input identity (two
    // applications may intentionally reuse the same feature input). This map
    // is placement-instance state and never enters model serialization.
    auto dropoutApplicationIndices = std::make_shared<std::unordered_map<uint64_t, uint32_t>>();
    auto dropoutApplicationIndicesMutex = std::make_shared<std::mutex>();

    return DynamicExpression(
        std::move(expectedInputNames),
        {"feature_output"},
        [apiLayerId,
         hasBias,
         preserveInputPrefixDimensions,
         placement,
         weightsDataType,
         computeDataType,
         outputDataType,
         activation = std::move(activationClone),
         packedRowCapacity,
         applicationMaxValuesPerRow = std::move(applicationMaxValuesPerRow),
         rowPartitionInputName = std::move(rowPartitionInputName),
         rowPartitionDescriptor,
         epilogue,
         epilogueAuxInputNames = std::move(epilogueAuxInputNames),
         outputDropoutProbability,
         outputDropoutSeed,
         dropoutApplicationIndices,
         dropoutApplicationIndicesMutex,
         useResidual](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            Tensor featureInputTensor = inputs.at("feature_input");
            // A CustomLayer expression is built independently for each physical
            // application. The outer stamp establishes placement B; this application's
            // actual values capacity selects its own semantic M. Never reuse the
            // first application's B*M or RowPartitionDescriptor here.
            std::optional<uint64_t> applicationPackedRowCapacity = packedRowCapacity;
            std::optional<ThorImplementation::RowPartitionDescriptor> applicationPartition = rowPartitionDescriptor;
            if (rowPartitionDescriptor.has_value()) {
                const auto& dims = featureInputTensor.getDimensions();
                const uint64_t b = rowPartitionDescriptor->getBatchSize();
                if (dims.size() != 2 || dims.front() == 0 || b == 0) {
                    throw std::runtime_error("Ragged FullyConnected application has invalid physical values dimensions.");
                }
                // Select semantic M from the authored applications, not by
                // treating the physical packed extent as model architecture.
                const auto mIt = std::find_if(applicationMaxValuesPerRow.begin(), applicationMaxValuesPerRow.end(),
                    [&](uint64_t semanticM) {
                        return semanticM != 0 && semanticM <= std::numeric_limits<uint64_t>::max() / b &&
                               b * semanticM == dims.front();
                    });
                if (mIt == applicationMaxValuesPerRow.end()) {
                    throw std::runtime_error("Ragged FullyConnected application has an unrecognized physical B*M capacity.");
                }
                const uint64_t m = *mIt;
                applicationPackedRowCapacity = dims.front();
                applicationPartition.emplace(b, dims.front(), rowPartitionDescriptor->getOffsetsDataType(), m);
            }
            std::optional<Tensor> rowPartitionTensor;
            const bool needsDeviceActiveExtent =
                hasBias || activation != nullptr || epilogue.has_value() ||
                outputDropoutProbability > 0.0f || useResidual;
            if (rowPartitionInputName.has_value()) {
                if (!applicationPartition.has_value()) {
                    throw std::runtime_error("Ragged FullyConnected is missing its logical row-partition descriptor.");
                }
                rowPartitionTensor = inputs.at(rowPartitionInputName.value());
                if (rowPartitionTensor->getPlacement() != placement) {
                    throw std::runtime_error("Ragged FullyConnected row-partition carrier placement does not match the layer placement.");
                }
                // This factory is invoked independently for every CustomLayer application. The
                // The stamp supplies common placement B and offsets dtype,
                // while this invocation reconstructs its own authored M and
                // current application's structural carrier.
                const auto& partition = applicationPartition.value();
                const auto& carrierDimensions = rowPartitionTensor->getDimensions();
                const bool activeCountCarrier =
                    rowPartitionTensor->getDataType() == partition.getOffsetsDataType() &&
                    carrierDimensions == std::vector<uint64_t>{1};
                const bool offsetsCarrier =
                    rowPartitionTensor->getDataType() == partition.getOffsetsDataType() &&
                    carrierDimensions == partition.getOffsetsDescriptor().getDimensions();
                // A HOST_EXTENT partition port is carried by the canonical source's
                // packed values. Its dtype and trailing dimensions need not match
                // this FullyConnected application's values after preceding layers.
                // Graph routing and the stamp-time row-partition descriptor establish
                // its identity; only its leading physical B*M extent is relevant.
                const bool hostExtentCarrier =
                    !carrierDimensions.empty() && applicationPackedRowCapacity.has_value() &&
                    carrierDimensions.front() == applicationPackedRowCapacity.value();
                if (!(activeCountCarrier || offsetsCarrier ||
                      (!needsDeviceActiveExtent && hostExtentCarrier))) {
                    throw std::runtime_error(
                        "Ragged FullyConnected application has incompatible physical row-partition carrier.");
                }
                if (!applicationPackedRowCapacity.has_value() ||
                    inputs.at("feature_input").getDimensions().empty() ||
                    inputs.at("feature_input").getDimensions().front() != applicationPackedRowCapacity.value()) {
                    throw std::runtime_error(
                        "Ragged FullyConnected application values capacity differs from B*M.");
                }
            } else if (applicationPartition.has_value()) {
                throw std::runtime_error("Dense FullyConnected unexpectedly received row-partition metadata.");
            }
            ThorImplementation::RaggedRuntimeExtentSource raggedRuntimeExtentSource =
                ThorImplementation::RaggedRuntimeExtentSource::DEVICE_OFFSETS;
            if (rowPartitionTensor.has_value()) {
                if (!needsDeviceActiveExtent) {
                    raggedRuntimeExtentSource = ThorImplementation::RaggedRuntimeExtentSource::HOST_EXTENT;
                } else {
                    const ThorImplementation::RowPartitionDescriptor& partition = applicationPartition.value();
                    const std::vector<uint64_t>& carrierDims = rowPartitionTensor->getDimensions();
                    if (carrierDims == partition.getOffsetsDescriptor().getDimensions() &&
                        rowPartitionTensor->getDataType() == partition.getOffsetsDataType()) {
                        // Training may strengthen a biased FC's physicalization to
                        // full offsets for its backward dBias segmented reduction.
                        // Reuse that representation for forward active-prefix
                        // kernels by reading offsets[B].
                        raggedRuntimeExtentSource = ThorImplementation::RaggedRuntimeExtentSource::DEVICE_OFFSETS;
                    } else if (carrierDims == std::vector<uint64_t>{1} &&
                               rowPartitionTensor->getDataType() == partition.getOffsetsDataType()) {
                        raggedRuntimeExtentSource = ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT;
                    } else {
                        throw std::runtime_error(
                            "Ragged FullyConnected device runtime extent requires managed [1] active count or [B+1] offsets.");
                    }
                }
            }
            const Tensor& wTensor = inputs.at("weights");
            if (wTensor.getDimensions().size() != 2) {
                throw std::runtime_error("FullyConnected weights tensor must be rank 2.");
            }
            if (wTensor.getDataType() != weightsDataType) {
                throw std::runtime_error("FullyConnected weights tensor dtype does not match weightsDataType.");
            }
            if (wTensor.getPlacement() != placement) {
                throw std::runtime_error("FullyConnected weights tensor placement does not match the layer placement.");
            }

            std::vector<uint64_t> featureInputDimensions = featureInputTensor.getDimensions();
            if (featureInputDimensions.size() < 2) {
                throw std::runtime_error(
                    "FullyConnected dynamic expression requires a feature input tensor with batch plus at least one feature dimension.");
            }
            if (featureInputTensor.getPlacement() != placement) {
                throw std::runtime_error("FullyConnected feature input placement does not match the layer placement.");
            }

            // Standard FullyConnected keeps the historical behavior: flatten every non-batch dimension into one
            // feature vector.  Tokenwise/sequence projections set preserveInputPrefixDimensions, treating only the
            // last logical dimension as features and folding [batch, ...prefix] into the matmul batch.  The output is
            // reshaped back to [batch, ...prefix, out_features], so language-model heads do not need a CustomLayer.
            const std::vector<uint64_t> originalFeatureInputDimensions = featureInputDimensions;
            std::vector<uint64_t> logicalFeatureInputDimensions;
            std::vector<uint64_t> runtimeFeatureOutputDimensions;
            if (preserveInputPrefixDimensions) {
                uint64_t flattenedItems = 1;
                for (uint32_t i = 0; i + 1 < featureInputDimensions.size(); ++i) {
                    if (featureInputDimensions[i] == 0) {
                        throw std::runtime_error("FullyConnected runtime prefix dimensions must be non-zero.");
                    }
                    if (flattenedItems > std::numeric_limits<uint64_t>::max() / featureInputDimensions[i]) {
                        throw std::runtime_error("FullyConnected flattened token count overflows uint64_t.");
                    }
                    flattenedItems *= featureInputDimensions[i];
                }
                const uint64_t inputFeatures = featureInputDimensions.back();
                if (inputFeatures == 0) {
                    throw std::runtime_error("FullyConnected runtime feature dimension must be non-zero.");
                }
                logicalFeatureInputDimensions = {flattenedItems, inputFeatures};
                runtimeFeatureOutputDimensions = featureInputDimensions;
                runtimeFeatureOutputDimensions.back() = wTensor.getDimensions()[1];
            } else {
                const uint64_t batchSize = featureInputDimensions[0];
                if (batchSize == 0) {
                    throw std::runtime_error("FullyConnected runtime batch dimension must be non-zero.");
                }
                uint64_t flattenedFeatures = 1;
                for (uint32_t i = 1; i < featureInputDimensions.size(); ++i) {
                    if (featureInputDimensions[i] == 0) {
                        throw std::runtime_error("FullyConnected runtime feature dimensions must be non-zero.");
                    }
                    if (flattenedFeatures > std::numeric_limits<uint64_t>::max() / featureInputDimensions[i]) {
                        throw std::runtime_error("FullyConnected flattened feature count overflows uint64_t.");
                    }
                    flattenedFeatures *= featureInputDimensions[i];
                }
                logicalFeatureInputDimensions = {batchSize, flattenedFeatures};
                runtimeFeatureOutputDimensions = {batchSize, wTensor.getDimensions()[1]};
            }

            if (logicalFeatureInputDimensions.size() != 2) {
                throw std::runtime_error("FullyConnected logical feature input tensor must be rank 2 after flattening.");
            }
            if (logicalFeatureInputDimensions[0] == 0 || logicalFeatureInputDimensions[1] == 0) {
                throw std::runtime_error("FullyConnected logical feature input tensor dimensions must be non-zero.");
            }
            if (logicalFeatureInputDimensions[1] != wTensor.getDimensions()[0]) {
                throw std::runtime_error(
                    "FullyConnected#" + std::to_string(apiLayerId) +
                    " input feature count does not match weights rows: preserveInputPrefixDimensions=" +
                    std::string(preserveInputPrefixDimensions ? "true" : "false") +
                    ", physical_input_dimensions=" + dimensionsString(originalFeatureInputDimensions) +
                    ", logical_matmul_input_dimensions=" + dimensionsString(logicalFeatureInputDimensions) +
                    ", weights_dimensions=" + dimensionsString(wTensor.getDimensions()) + ".");
            }
            if (outputs.contains("feature_output")) {
                const Tensor& featureOutputTensor = outputs.at("feature_output");
                if (featureOutputTensor.getDimensions() != runtimeFeatureOutputDimensions) {
                    throw std::runtime_error("FullyConnected feature output tensor dimensions are incompatible with the matmul output.");
                }
                if (featureOutputTensor.getDataType() != outputDataType) {
                    throw std::runtime_error("FullyConnected feature output tensor dtype does not match outputDataType.");
                }
                if (featureOutputTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected feature output tensor placement does not match the layer placement.");
                }
            }

            std::optional<Tensor> denseBatchValidityTensor;
            if (!rowPartitionTensor.has_value()) {
                denseBatchValidityTensor = inputs.at(Thor::BATCH_VALIDITY_MASK_NAME);
                if (denseBatchValidityTensor->getDataType() != DataType::FP32) {
                    throw std::runtime_error("Dense FullyConnected batch-validity mask must be FP32.");
                }
                if (denseBatchValidityTensor->getPlacement() != placement) {
                    throw std::runtime_error("Dense FullyConnected batch-validity mask placement does not match the layer placement.");
                }
                std::vector<uint64_t> expectedValidityDimensions;
                if (preserveInputPrefixDimensions) {
                    expectedValidityDimensions = originalFeatureInputDimensions;
                    expectedValidityDimensions.back() = 1;
                } else {
                    expectedValidityDimensions.assign(originalFeatureInputDimensions.size(), 1);
                    expectedValidityDimensions.front() = originalFeatureInputDimensions.front();
                }
                if (denseBatchValidityTensor->getDimensions() != expectedValidityDimensions) {
                    throw std::runtime_error("Dense FullyConnected batch-validity mask has unexpected dimensions.");
                }
            }

            auto denseValidityForDimensions = [&](const std::vector<uint64_t>& dimensions) -> Expression {
                THOR_THROW_IF_FALSE(denseBatchValidityTensor.has_value());
                THOR_THROW_IF_FALSE(!dimensions.empty());

                std::vector<uint64_t> maskDimensions;
                if (preserveInputPrefixDimensions) {
                    maskDimensions = dimensions;
                    maskDimensions.back() = 1;
                } else {
                    maskDimensions.assign(dimensions.size(), 1);
                    maskDimensions.front() = dimensions.front();
                }

                uint64_t maskElements = 1;
                for (uint64_t dim : maskDimensions) {
                    THOR_THROW_IF_FALSE(dim != 0);
                    THOR_THROW_IF_FALSE(maskElements <= std::numeric_limits<uint64_t>::max() / dim);
                    maskElements *= dim;
                }
                if (maskElements != denseBatchValidityTensor->getTotalNumElements()) {
                    throw std::runtime_error(
                        "Dense FullyConnected batch-validity mask cannot represent the requested effective geometry.");
                }

                Expression validity = Expression::input(
                    Thor::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
                if (denseBatchValidityTensor->getDimensions() != maskDimensions) {
                    validity = validity.reshape(maskDimensions);
                }
                return validity;
            };

            auto sanitizeDenseBatchInput = [&](Expression value, const std::vector<uint64_t>& dimensions) -> Expression {
                if (!denseBatchValidityTensor.has_value()) {
                    return value;
                }
                Expression validity = denseValidityForDimensions(dimensions);
                // Selection is intentional here. Multiplication by a 0/1 mask is
                // not sufficient because IEEE NaN * 0 remains NaN and can poison
                // a parameter-gradient reduction from inactive physical rows.
                return Expression::where(
                    validity > Expression::constantScalar(0.0),
                    value,
                    Expression::constantScalar(0.0));
            };

            auto fin = Expression::input("feature_input", featureInputTensor.getDataType(), featureInputTensor.getDataType());
            fin = sanitizeDenseBatchInput(fin, originalFeatureInputDimensions);
            if (originalFeatureInputDimensions != logicalFeatureInputDimensions) {
                fin = fin.reshape(logicalFeatureInputDimensions);
            }
            if (rowPartitionTensor.has_value()) {
                if (!applicationPackedRowCapacity.has_value() || logicalFeatureInputDimensions[0] != applicationPackedRowCapacity.value()) {
                    throw std::runtime_error("Ragged FullyConnected packed capacity does not match its flattened values rows.");
                }
                const ThorImplementation::RowPartitionDescriptor& partition = applicationPartition.value();
                const uint64_t raggedBatchSize = partition.getBatchSize();
                Expression offsets = Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT,
                                                       partition.getOffsetsDataType(),
                                                       partition.getOffsetsDataType());
                fin = fin.withRaggedRuntimeExtent(offsets,
                                                  raggedBatchSize,
                                                  applicationPackedRowCapacity.value(),
                                                  logicalFeatureInputDimensions[1],
                                                  raggedRuntimeExtentSource);
            }
            auto w = Expression::input("weights", weightsDataType, weightsDataType);

            // [batch, in_features] @ [in_features, out_features]
            Expression fout = Expression::matmul(fin, w, false, false, computeDataType, outputDataType, applicationPackedRowCapacity);

            if (hasBias) {
                const Tensor& bTensor = inputs.at("biases");
                if (bTensor.getDimensions().size() != 1 || bTensor.getDimensions()[0] != wTensor.getDimensions()[1]) {
                    throw std::runtime_error("FullyConnected biases tensor dimensions are incompatible with the weights tensor.");
                }
                if (bTensor.getDataType() != outputDataType) {
                    throw std::runtime_error("FullyConnected biases tensor dtype must match outputDataType.");
                }
                if (bTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected biases tensor placement does not match the layer placement.");
                }

                auto b = Expression::input("biases", outputDataType, outputDataType);

                // Broadcast [out_features] over batch.
                fout = fout + b;
            }

            if (activation != nullptr) {
                fout = activation->toExpression(fout);
            }

            const std::vector<uint64_t> matmulOutputDimensions = {logicalFeatureInputDimensions[0], wTensor.getDimensions()[1]};
            for (const std::string& auxInputName : epilogueAuxInputNames) {
                const Tensor& auxTensor = inputs.at(auxInputName);
                const std::vector<uint64_t>& expectedAuxShape = epilogue.has_value() ? runtimeFeatureOutputDimensions : matmulOutputDimensions;
                if (auxTensor.getDimensions() != expectedAuxShape) {
                    throw std::runtime_error("FullyConnected epilogue auxiliary input '" + auxInputName +
                                             "' shape must match the fully connected feature output shape. expected=" +
                                             dimensionsString(expectedAuxShape) + ", actual=" +
                                             dimensionsString(auxTensor.getDimensions()) + ".");
                }
                if (auxTensor.getDataType() != outputDataType) {
                    throw std::runtime_error("FullyConnected epilogue auxiliary input '" + auxInputName +
                                             "' dtype must match the fully connected feature output dtype.");
                }
                if (auxTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected epilogue auxiliary input placement does not match the layer placement.");
                }
            }

            if (useResidual) {
                const Tensor& residualTensor = inputs.at(FC_RESIDUAL_INPUT_NAME);
                if (residualTensor.getDimensions() != runtimeFeatureOutputDimensions) {
                    throw std::runtime_error("FullyConnected residual input shape must match the feature output shape.");
                }
                if (residualTensor.getDataType() != outputDataType) {
                    throw std::runtime_error("FullyConnected residual input dtype must match outputDataType.");
                }
                if (residualTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected residual input placement does not match the layer placement.");
                }
            }

            const Expression preOutputPostOp = fout;
            auto buildOutputBranch = [&](bool enableOutputDropout) -> Expression {
                Expression branch = preOutputPostOp;

                // Canonical composed order: affine -> activation -> custom epilogue
                // -> output dropout -> residual. An epilogue transforms the projected
                // branch, never the skip connection. This preserves the established
                // epilogue-only and dropout/residual-only paths while allowing both.
                if (epilogue.has_value()) {
                    // Keep the entire epilogue in the folded matmul geometry. In training, applying a rank-3
                    // residual epilogue after first reshaping the primary result to [B, ...prefix, O] causes the
                    // same upstream gradient tensor to participate in both the public rank-3 domain and the
                    // folded [B*prefix, O] domain in one backward fused equation. Flatten each public-shape
                    // auxiliary epilogue input instead and restore the public shape only once after the epilogue.
                    Expression effectiveEpilogue = epilogue.value();
                    for (const std::string& auxInputName : epilogueAuxInputNames) {
                        const ExpressionInputDataTypes inputDataTypes = expressionInputDataTypes(effectiveEpilogue, auxInputName);
                        Expression physicalAuxInput =
                            Expression::input(auxInputName, inputDataTypes.computeDataType, inputDataTypes.outputDataType);
                        physicalAuxInput = sanitizeDenseBatchInput(
                            physicalAuxInput, runtimeFeatureOutputDimensions);
                        if (runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                            physicalAuxInput = physicalAuxInput.reshape(matmulOutputDimensions);
                        }
                        if (rowPartitionTensor.has_value()) {
                            if (!applicationPackedRowCapacity.has_value() || matmulOutputDimensions.size() != 2 ||
                                matmulOutputDimensions[0] != applicationPackedRowCapacity.value()) {
                                throw std::runtime_error(
                                    "Ragged FullyConnected epilogue auxiliary geometry does not match its packed row capacity.");
                            }
                            const ThorImplementation::RowPartitionDescriptor& partition = applicationPartition.value();
                            Expression offsets = Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT,
                                                                   partition.getOffsetsDataType(),
                                                                   partition.getOffsetsDataType());
                            physicalAuxInput = physicalAuxInput.withRaggedRuntimeExtent(
                                offsets,
                                partition.getBatchSize(),
                                applicationPackedRowCapacity.value(),
                                matmulOutputDimensions[1],
                                raggedRuntimeExtentSource);
                        }
                        effectiveEpilogue = effectiveEpilogue.substituteInput(auxInputName, physicalAuxInput);
                    }
                    if (rowPartitionTensor.has_value()) {
                        if (!applicationPackedRowCapacity.has_value() || matmulOutputDimensions.size() != 2 ||
                            matmulOutputDimensions[0] != applicationPackedRowCapacity.value()) {
                            throw std::runtime_error(
                                "Ragged FullyConnected epilogue primary geometry does not match its packed row capacity.");
                        }
                        const ThorImplementation::RowPartitionDescriptor& partition = applicationPartition.value();
                        Expression offsets = Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT,
                                                               partition.getOffsetsDataType(),
                                                               partition.getOffsetsDataType());
                        branch = branch.withRaggedRuntimeExtent(offsets,
                                                                partition.getBatchSize(),
                                                                applicationPackedRowCapacity.value(),
                                                                matmulOutputDimensions[1],
                                                                raggedRuntimeExtentSource);
                    }
                    branch = FullyConnected::applyEpilogue(branch, effectiveEpilogue);
                }


                std::optional<Expression> flattenedResidual;
                if (useResidual) {
                    flattenedResidual = Expression::input(FC_RESIDUAL_INPUT_NAME, outputDataType, outputDataType);
                    flattenedResidual = sanitizeDenseBatchInput(
                        flattenedResidual.value(), runtimeFeatureOutputDimensions);
                    if (runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                        flattenedResidual = flattenedResidual->reshape(matmulOutputDimensions);
                    }
                }

                if (enableOutputDropout && outputDropoutProbability > 0.0f) {
                    const bool ragged = rowPartitionTensor.has_value();
                    DataType offsetsDataType = DataType::UINT32;
                    uint64_t raggedBatchSize = 0;
                    uint64_t featuresPerValue = 0;
                    if (ragged) {
                        if (!applicationPartition.has_value()) {
                            throw std::runtime_error(
                                "Ragged FullyConnected output dropout is missing its logical row-partition descriptor.");
                        }
                        offsetsDataType = applicationPartition->getOffsetsDataType();
                        raggedBatchSize = applicationPartition->getBatchSize();
                        featuresPerValue = matmulOutputDimensions[1];
                    }
                    ThorImplementation::CudaKernelExpression outputDropout =
                        ThorImplementation::makeDropOutPostOpKernel(outputDataType,
                                                                  outputDropoutProbability,
                                                                  useResidual,
                                                                  ragged,
                                                                  offsetsDataType,
                                                                  raggedBatchSize,
                                                                  featuresPerValue,
                                                                  "FullyConnected output",
                                                                  raggedRuntimeExtentSource);
                    std::unordered_map<std::string, Expression> dropoutInputs{
                        {"projected", branch},
                        {"seed", Expression::tensorRuntimeScalar(
                                     FC_OUTPUT_DROPOUT_SEED_INPUT_NAME, DataType::INT64, DataType::INT64)},
                        {"sequence", Expression::tensorRuntimeScalar(
                                         FC_OUTPUT_DROPOUT_SEQUENCE_INPUT_NAME, DataType::INT64, DataType::INT64)},
                    };
                    if (flattenedResidual.has_value()) {
                        dropoutInputs.emplace("residual", flattenedResidual.value());
                    }
                    if (ragged) {
                        const char* partitionInputName =
                            raggedRuntimeExtentSource == ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT
                                ? "active_count"
                                : "offsets";
                        dropoutInputs.emplace(
                            partitionInputName,
                            Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT, offsetsDataType, offsetsDataType));
                    }
                    ThorImplementation::Outputs dropoutOutputs = outputDropout.apply(dropoutInputs);
                    const std::vector<std::string> dropoutOutputNames = dropoutOutputs.outputNames();
                    if (dropoutOutputNames.size() != 1 || dropoutOutputNames.front() != "output") {
                        throw std::logic_error("FullyConnected output-dropout kernel produced an unexpected output interface.");
                    }
                    branch = dropoutOutputs.outputExpression("output");
                } else if (flattenedResidual.has_value()) {
                    // Keep the residual add directly adjacent to the affine/activation branch. For the common
                    // Transformer final-projection case (no activation), EquationCompiler can lower this into
                    // the existing cuBLASLt GEMM beta/residual path.
                    branch = branch + flattenedResidual.value();
                }


                if (runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                    branch = branch.reshape(runtimeFeatureOutputDimensions);
                }

                if (denseBatchValidityTensor.has_value()) {
                    Expression outputValidity = denseValidityForDimensions(runtimeFeatureOutputDimensions);
                    // Mask the public branch as well as its inputs. Besides making
                    // inactive forward rows deterministic, WHERE autodiff selects an
                    // exact zero incoming gradient for inactive rows. This prevents a
                    // downstream poisoned tail from reaching matmul/bias/epilogue
                    // gradient reductions as NaN * 0.
                    branch = Expression::where(
                        outputValidity > Expression::constantScalar(0.0),
                        branch,
                        Expression::constantScalar(0.0));
                }

                branch = branch.withOutputDType(outputDataType);

                if (rowPartitionTensor.has_value()) {
                    if (!applicationPackedRowCapacity.has_value() || runtimeFeatureOutputDimensions.empty() ||
                        runtimeFeatureOutputDimensions[0] != applicationPackedRowCapacity.value()) {
                        throw std::runtime_error(
                            "Ragged FullyConnected output geometry does not match its packed row capacity.");
                    }
                    uint64_t outputElementsPerPackedValue = 1;
                    for (size_t axis = 1; axis < runtimeFeatureOutputDimensions.size(); ++axis) {
                        if (runtimeFeatureOutputDimensions[axis] == 0 ||
                            outputElementsPerPackedValue >
                                std::numeric_limits<uint64_t>::max() / runtimeFeatureOutputDimensions[axis]) {
                            throw std::runtime_error("Ragged FullyConnected output row width overflows uint64_t.");
                        }
                        outputElementsPerPackedValue *= runtimeFeatureOutputDimensions[axis];
                    }
                    const ThorImplementation::RowPartitionDescriptor& partition = applicationPartition.value();
                    Expression offsets = Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT,
                                                           partition.getOffsetsDataType(),
                                                           partition.getOffsetsDataType());
                    branch = branch.withRaggedRuntimeExtent(offsets,
                                                            partition.getBatchSize(),
                                                            applicationPackedRowCapacity.value(),
                                                            outputElementsPerPackedValue,
                                                            raggedRuntimeExtentSource);
                }
                return branch;
            };

            auto expressionOutputs = Expression::outputs({{"feature_output", buildOutputBranch(outputDropoutProbability > 0.0f)}});
            std::shared_ptr<FusedEquation> evaluationEquation;
            if (outputDropoutProbability > 0.0f) {
                auto evaluationOutputs = Expression::outputs({{"feature_output", buildOutputBranch(false)}});
                evaluationEquation = std::make_shared<FusedEquation>(
                    FusedEquation::compile(evaluationOutputs.physicalOutputs(), placement.getDeviceNum()));
            }

            DynamicExpression::TensorMap stampInputs = inputs;
            DynamicExpression::TensorMap preForwardOnlyInputs;
            std::function<void(Stream&)> preForwardHook;
            if (rowPartitionInputName.has_value()) {
                stampInputs.erase(rowPartitionInputName.value());
                stampInputs.emplace(RAGGED_ROW_PARTITION_EXPRESSION_INPUT, rowPartitionTensor.value());
                preForwardOnlyInputs.emplace(rowPartitionInputName.value(), rowPartitionTensor.value());
                // The public logical row-partition port remains a non-differentiable structural dependency of
                // RaggedFullyConnected. The private binding receives the placement-selected physical carrier:
                // authoritative host metadata for packed bucket selection plus, when requested, the managed [1]
                // active-count payload for device valuewise stages.
                preForwardHook = [](Stream&) {};
            }

            std::unordered_map<std::string, TensorScalarBinding> tensorScalarInputs;
            if (outputDropoutProbability > 0.0f) {
                // Each stamped application has independent Philox state. A
                // discovery-only prepare() is not executable and does not reserve
                // a stream; the actual output uniquely identifies an application.
                int64_t applicationSeed = outputDropoutSeed;
                if (rowPartitionDescriptor.has_value() && applicationMaxValuesPerRow.size() > 1 &&
                    outputs.contains("feature_output")) {
                    const uint64_t tensorId = outputs.at("feature_output").getTensorId();
                    if (tensorId == 0) {
                        throw std::runtime_error("FullyConnected output dropout requires a physical application output identity.");
                    }
                    uint32_t applicationIndex;
                    {
                        std::lock_guard<std::mutex> guard(*dropoutApplicationIndicesMutex);
                        auto [it, inserted] = dropoutApplicationIndices->try_emplace(
                            tensorId, static_cast<uint32_t>(dropoutApplicationIndices->size()));
                        (void)inserted;
                        applicationIndex = it->second;
                    }
                    if (applicationIndex >= applicationMaxValuesPerRow.size()) {
                        throw std::runtime_error("FullyConnected output dropout discovered more physical applications than were authored.");
                    }
                    if (applicationIndex != 0) {
                        // Derive nonidentical reproducible substream seeds from
                        // the serialized seed without saving placement state.
                        constexpr uint64_t kStreamStride = 0x9E3779B97F4A7C15ULL;
                        const uint64_t mixed = static_cast<uint64_t>(outputDropoutSeed) ^
                                               (kStreamStride * static_cast<uint64_t>(applicationIndex));
                        applicationSeed = static_cast<int64_t>(mixed &
                            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
                    }
                }
                auto outputDropoutState = std::make_shared<ThorImplementation::DropOutRuntimeState>(
                    applicationSeed, 0, "FullyConnected output");
                outputDropoutState->setSequenceAdvance(1);
                tensorScalarInputs[FC_OUTPUT_DROPOUT_SEED_INPUT_NAME] =
                    outputDropoutState->seedBinding(featureInputTensor.getPlacement());
                tensorScalarInputs[FC_OUTPUT_DROPOUT_SEQUENCE_INPUT_NAME] =
                    outputDropoutState->sequenceBinding(featureInputTensor.getPlacement());
                const std::function<void(Stream&)> previousHook = preForwardHook;
                preForwardHook = [previousHook, outputDropoutState](Stream& runStream) {
                    if (previousHook) previousHook(runStream);
                    outputDropoutState->uploadForForward(runStream);
                };
            }

            DynamicExpressionBuild build{
                .equation = std::make_shared<FusedEquation>(
                    FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
                .stamp_inputs = std::move(stampInputs),
                .tensor_scalar_inputs = std::move(tensorScalarInputs),
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = std::move(preForwardHook),
                .serialized_definition = nullptr,
                .execution_variants = {},
                .evaluation_variant_id = std::nullopt,
                .pre_forward_only_inputs = std::move(preForwardOnlyInputs),
            };
            if (evaluationEquation) {
                build.execution_variants.emplace(
                    FC_EVALUATION_VARIANT,
                    DynamicExpressionVariant{
                        .equation = std::move(evaluationEquation),
                        .tensor_scalar_inputs = {},
                        .pre_forward_hook = rowPartitionInputName.has_value() ? std::function<void(Stream&)>([](Stream&) {})
                                                                            : std::function<void(Stream&)>{},
                        .supports_backward = true,
                    });
                build.evaluation_variant_id = FC_EVALUATION_VARIANT;
            }
            return build;
        });
}

}  // namespace

bool FullyConnected::isFullyConnectedFloatingDataType(DataType dataType) {
    return Thor::isFullyConnectedFloatingDataType(dataType);
}

std::string FullyConnected::dataTypeName(DataType dataType) {
    return fullyConnectedDataTypeName(dataType);
}

uint64_t FullyConnected::checkedFeatureCount(const std::vector<uint64_t>& dimensions, const std::string& what) {
    if (dimensions.empty()) {
        throw std::invalid_argument("FullyConnected " + what + " must have at least one feature dimension.");
    }

    uint64_t featureCount = 1;
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::invalid_argument("FullyConnected " + what + " dimensions must be non-zero.");
        }
        if (featureCount > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::invalid_argument("FullyConnected " + what + " feature count overflows uint64_t.");
        }
        featureCount *= dim;
    }

    if (featureCount > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("FullyConnected " + what + " feature count exceeds the int32 cuBLASLt interface limit.");
    }

    return featureCount;
}
uint64_t FullyConnected::checkedInputFeatureCount(const std::vector<uint64_t>& dimensions,
                                                   bool preservePrefixDimensions,
                                                   const std::string& what) {
    if (!preservePrefixDimensions) {
        return checkedFeatureCount(dimensions, what);
    }
    if (dimensions.empty()) {
        throw std::invalid_argument("FullyConnected " + what + " must have at least one feature dimension.");
    }
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::invalid_argument("FullyConnected " + what + " dimensions must be non-zero.");
        }
    }
    const uint64_t featureCount = dimensions.back();
    if (featureCount > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("FullyConnected " + what + " feature count exceeds the int32 cuBLASLt interface limit.");
    }
    return featureCount;
}

std::vector<uint64_t> FullyConnected::fullyConnectedOutputDimensions(const std::vector<uint64_t>& inputDimensions,
                                                                     uint32_t numOutputFeatures,
                                                                     bool preservePrefixDimensions) {
    if (!preservePrefixDimensions) {
        return {numOutputFeatures};
    }
    if (inputDimensions.empty()) {
        throw std::invalid_argument("FullyConnected input dimensions must be non-empty.");
    }
    std::vector<uint64_t> outputDimensions = inputDimensions;
    outputDimensions.back() = numOutputFeatures;
    return outputDimensions;
}


void FullyConnected::verifyFullyConnectedDataType(DataType dataType, const std::string& what) {
    if (!isFullyConnectedFloatingDataType(dataType)) {
        throw std::invalid_argument("FullyConnected " + what + " must be one of fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32. Got " +
                                    dataTypeName(dataType) + ".");
    }
}

DataType FullyConnected::defaultFullyConnectedComputeDataType(DataType inputDataType, DataType weightsDataType, DataType outputDataType) {
    // cuBLASLt exposes FP8 GEMM operation types only with strict FP32 compute.
    // Treat FP8 appearing anywhere in the storage plan as an FP32-compute plan;
    // non-FP8 FullyConnected keeps the historical feature-input default (with
    // explicit TF32 still available for FP32 storage).
    if (isFullyConnectedFp8DataType(inputDataType) || isFullyConnectedFp8DataType(weightsDataType) ||
        isFullyConnectedFp8DataType(outputDataType)) {
        return DataType::FP32;
    }
    return inputDataType;
}

void FullyConnected::verifyFullyConnectedComputeDataType(DataType dataType) {
    if (!cublasLtComputeTypeForFullyConnected(dataType).has_value()) {
        throw std::invalid_argument(
            "FullyConnected computeDataType must be fp32, tf32, fp16, or bf16 for Thor's current cuBLASLt floating GEMM path. Got " +
            dataTypeName(dataType) + ".");
    }
}

void FullyConnected::validateEpilogueAuxInputName(const std::string& inputName) {
    if (inputName.empty()) {
        throw std::invalid_argument("FullyConnected epilogue auxiliary input name cannot be empty.");
    }
    if (inputName.rfind("__", 0) == 0) {
        throw std::invalid_argument("FullyConnected epilogue auxiliary input names cannot start with __: " + inputName + ".");
    }
    static const std::set<std::string> reservedNames = {
        "feature_input",
        "feature_output",
        "weights",
        "biases",
        FC_RESIDUAL_INPUT_NAME,
        ThorImplementation::RaggedFullyConnected::ROW_PARTITION_INPUT_NAME,
        epilogueInputName(),
        epilogueOutputName(),
    };
    if (reservedNames.contains(inputName)) {
        throw std::invalid_argument("FullyConnected epilogue auxiliary input name is reserved: " + inputName + ".");
    }
}

std::vector<std::string> FullyConnected::epilogueAuxInputNames() const {
    std::vector<std::string> names;
    names.reserve(epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)tensor;
        if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
    }
    return names;
}

std::vector<Tensor> FullyConnected::getFeatureInputs() const {
    if (!raggedFeatureInputs.empty()) {
        std::vector<Tensor> inputs;
        const uint32_t auxiliaryCount = static_cast<uint32_t>(epilogueAuxInputNames().size());
        inputs.reserve(raggedFeatureInputs.size() * (2 + (residualInput.has_value() ? 1 : 0) + auxiliaryCount));
        for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
            const RaggedTensor& ragged = raggedFeatureInputs[i];
            inputs.push_back(ragged.getValues());
            inputs.push_back(ragged.getOffsets());
            if (residualInput.has_value()) {
                const auto residual = getRaggedResidualInput(i);
                THOR_THROW_IF_FALSE(residual.has_value());
                inputs.push_back(residual->getValues());
            }
            for (uint32_t auxiliary = 0; auxiliary < auxiliaryCount; ++auxiliary) {
                inputs.push_back(epilogueInputBindings.at(i * auxiliaryCount + auxiliary).second);
            }
        }
        return inputs;
    }

    std::vector<Tensor> inputs = featureInputs;
    inputs.reserve(inputs.size() + (residualInput.has_value() ? 1 : 0) + epilogueInputBindings.size());
    if (residualInput.has_value()) inputs.push_back(residualInput.value());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        inputs.push_back(tensor);
    }
    return inputs;
}

std::vector<uint32_t> FullyConnected::inputPortIndicesForTensor(Tensor tensor) const {
    std::vector<uint32_t> ports;
    if (!raggedFeatureInputs.empty()) {
        const uint32_t auxiliaryCount = static_cast<uint32_t>(epilogueAuxInputNames().size());
        const uint32_t basePorts = residualInput.has_value() ? 3u : 2u;
        const uint32_t stride = basePorts + auxiliaryCount;
        for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
            if (tensor.getOriginalId() == raggedFeatureInputs[i].getValues().getOriginalId())
                ports.push_back(i * stride);
            if (tensor.getOriginalId() == raggedFeatureInputs[i].getOffsets().getOriginalId())
                ports.push_back(i * stride + 1);
            if (residualInput.has_value()) {
                const auto residual = getRaggedResidualInput(i);
                THOR_THROW_IF_FALSE(residual.has_value());
                if (tensor.getOriginalId() == residual->getValues().getOriginalId())
                    ports.push_back(i * stride + 2);
            }
            for (uint32_t auxiliary = 0; auxiliary < auxiliaryCount; ++auxiliary) {
                if (tensor.getOriginalId() == epilogueInputBindings.at(i * auxiliaryCount + auxiliary).second.getOriginalId())
                    ports.push_back(i * stride + basePorts + auxiliary);
            }
        }
        return ports;
    }

    if (!featureInputs.empty() && tensor.getOriginalId() == featureInputs[0].getOriginalId()) {
        ports.push_back(0);
    }
    uint32_t nextPort = 1;
    if (residualInput.has_value()) {
        if (tensor.getOriginalId() == residualInput->getOriginalId()) ports.push_back(nextPort);
        ++nextPort;
    }
    for (uint32_t i = 0; i < epilogueInputBindings.size(); ++i) {
        if (tensor.getOriginalId() == epilogueInputBindings[i].second.getOriginalId()) {
            ports.push_back(nextPort + i);
        }
    }
    return ports;
}

std::vector<Tensor> FullyConnected::getOutputsFromInput(Tensor inputTensor) {
    if (!raggedFeatureInputs.empty()) {
        if (inputPortIndicesForTensor(inputTensor).empty()) {
            throw std::runtime_error("FullyConnected received an unknown ragged input tensor.");
        }
        std::vector<Tensor> readyOutputs;
        const uint32_t auxiliaryCount = static_cast<uint32_t>(epilogueAuxInputNames().size());
        const uint32_t basePorts = residualInput.has_value() ? 3u : 2u;
        const uint32_t stride = basePorts + auxiliaryCount;
        for (uint32_t applicationIndex = 0; applicationIndex < raggedFeatureInputs.size(); ++applicationIndex) {
            const uint32_t valuesPort = applicationIndex * stride;
            if (emittedRaggedOutputApplications.contains(applicationIndex)) continue;
            bool ready = true;
            for (uint32_t port = 0; port < stride; ++port) {
                if (!connectedInputPortIndices.contains(valuesPort + port)) {
                    ready = false;
                    break;
                }
            }
            if (!ready) continue;
            THOR_THROW_IF_FALSE(applicationIndex < featureOutputs.size());
            emittedRaggedOutputApplications.insert(applicationIndex);
            readyOutputs.push_back(featureOutputs[applicationIndex]);
        }
        return readyOutputs;
    }

    if (epilogueInputBindings.empty() && !residualInput.has_value()) {
        return {getFeatureOutput(inputTensor)};
    }

    (void)getFeatureOutput(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected) return {};
    const uint32_t requiredInputPorts =
        static_cast<uint32_t>(1 + (residualInput.has_value() ? 1 : 0) + epilogueInputBindings.size());
    if (connectedInputPortIndices.size() != requiredInputPorts) return {};

    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs[0]};
}

void FullyConnected::informThatInputConnectionMade(Tensor inputTensor) {
    if (!raggedFeatureInputs.empty()) {
        std::vector<uint32_t> ports = inputPortIndicesForTensor(inputTensor);
        if (ports.empty()) {
            throw std::runtime_error("FullyConnected informed of connection for unknown ragged input tensor.");
        }
        uint32_t& cursor = nextTraversalInputCursorByTensorOriginalId[inputTensor.getOriginalId()];
        connectedInputPortIndices.insert(ports[cursor % ports.size()]);
        ++cursor;
        return;
    }

    if (epilogueInputBindings.empty() && !residualInput.has_value()) {
        return;
    }
    std::vector<uint32_t> ports = inputPortIndicesForTensor(inputTensor);
    if (ports.empty()) {
        throw std::runtime_error("FullyConnected informed of connection for unknown input tensor.");
    }
    for (uint32_t port : ports) {
        connectedInputPortIndices.insert(port);
    }
}

void FullyConnected::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedRaggedOutputApplications.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
    nextTraversalInputCursorByTensorOriginalId.clear();
}

int FullyConnected::getConnectionType(Tensor connectingTensor) const {
    if (!raggedFeatureInputs.empty() || !epilogueInputBindings.empty() || residualInput.has_value()) {
        std::vector<uint32_t> inputPorts = inputPortIndicesForTensor(connectingTensor);
        if (!inputPorts.empty()) {
            uint32_t& cursor = nextInputConnectionCursorByTensorOriginalId[connectingTensor.getOriginalId()];
            const uint32_t port = inputPorts[cursor % inputPorts.size()];
            ++cursor;
            return static_cast<int>(port);
        }
    } else {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return static_cast<int>(i);
        }
    }

    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        if (connectingTensor == featureOutputs[i])
            return static_cast<int>(i);
    }

    throw std::runtime_error("Tensor is not connected to this FullyConnected layer.");
}

FullyConnected FullyConnected::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(!_featureInputs.empty());
    THOR_THROW_IF_FALSE(_numOutputFeatures.has_value());
    if (!_hasBias.has_value())
        _hasBias = false;
    if (!_preserveInputPrefixDimensions.has_value())
        _preserveInputPrefixDimensions = !_raggedFeatureInputs.empty();
    if (_weightsInitializer == nullptr)
        _weightsInitializer = Glorot::Builder().build();
    if (_biasInitializer == nullptr)
        _biasInitializer = Glorot::Builder().build();
    if (!_activation && !_activationExplicitlyRemoved) {
        _activation = Gelu::Builder().build();
    } else if (_activation != nullptr) {
        _activation = std::dynamic_pointer_cast<Activation>(_activation->clone());
        if (_activation == nullptr) {
            throw std::runtime_error("FullyConnected activation clone did not produce an Activation.");
        }
    }
    if (!_weightsDataType.has_value())
        _weightsDataType = _featureInputs[0].getDataType();
    if (!_outputDataType.has_value())
        _outputDataType = _featureInputs[0].getDataType();
    if (!_computeDataType.has_value())
        _computeDataType = FullyConnected::defaultFullyConnectedComputeDataType(
            _featureInputs[0].getDataType(), _weightsDataType.value(), _outputDataType.value());
    if (!_outputDropoutProbability.has_value())
        _outputDropoutProbability = 0.0f;
    if (!_outputDropoutSeed.has_value()) {
        if (_outputDropoutProbability.value() > 0.0f) {
            std::random_device rd;
            const uint64_t high = static_cast<uint64_t>(rd()) << 32U;
            const uint64_t low = static_cast<uint64_t>(rd());
            _outputDropoutSeed =
                static_cast<int64_t>((high ^ low) & static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
        } else {
            _outputDropoutSeed = 0;
        }
    }

    verifyConfig();

    FullyConnected fullyConnected(_epilogue,
                                  _epilogueInputBindings,
                                  _raggedEpilogueInputBindings,
                                  _outputDropoutProbability.value(),
                                  _outputDropoutSeed.value(),
                                  _residualInput,
                                  _raggedResidualInput);
    fullyConnected.additionalRaggedResidualInputs = _additionalRaggedResidualInputs;

    fullyConnected.featureInputs = _featureInputs;
    fullyConnected.raggedFeatureInputs = _raggedFeatureInputs;
    fullyConnected.numOutputFeatures = _numOutputFeatures.value();

    fullyConnected.hasBias = _hasBias.value();
    fullyConnected.preserveInputPrefixDimensions = _preserveInputPrefixDimensions.value();
    if (_activation != nullptr)
        fullyConnected.activation = _activation;
    fullyConnected.weightsDataType = _weightsDataType.value();
    fullyConnected.computeDataType = _computeDataType.value();
    fullyConnected.outputDataType = _outputDataType.value();

    // Own parameter intent at the API layer. The stamped implementation layer is now the generic
    // CustomLayer, so there is no implementation FullyConnected class left to define parameters.
    std::shared_ptr<Initializer> weightsInitializer = _weightsInitializer->clone();
    std::shared_ptr<Initializer> biasInitializer = _hasBias.value() ? _biasInitializer->clone() : nullptr;
    const uint64_t inputFeatures = FullyConnected::checkedInputFeatureCount(
        _featureInputs.front().getDimensions(), _preserveInputPrefixDimensions.value(), "feature input");

    ParameterSpecification::Builder weightsParameterBuilder;
    weightsParameterBuilder.name("weights")
        .shape({inputFeatures, fullyConnected.numOutputFeatures})
        .dtype(fullyConnected.weightsDataType)
        .initializer(weightsInitializer)
        .trainable(true);
    if (_weightsOptimizer != nullptr)
        weightsParameterBuilder.optimizer(_weightsOptimizer);
    weightsParameterBuilder.constraints(_weightsConstraints);
    fullyConnected.addParameter(std::make_shared<ParameterSpecification>(weightsParameterBuilder.build()));

    if (fullyConnected.hasBias) {
        ParameterSpecification::Builder biasesParameterBuilder;
        biasesParameterBuilder.name("biases")
            .shape({fullyConnected.numOutputFeatures})
            .dtype(fullyConnected.outputDataType)
            .initializer(biasInitializer)
            .trainable(true);
        if (_biasesOptimizer != nullptr)
            biasesParameterBuilder.optimizer(_biasesOptimizer);
        biasesParameterBuilder.constraints(_biasesConstraints);
        fullyConnected.addParameter(std::make_shared<ParameterSpecification>(biasesParameterBuilder.build()));
    }

    fullyConnected.initialized = true;

    for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
        Tensor out(fullyConnected.outputDataType,
                   FullyConnected::fullyConnectedOutputDimensions(fullyConnected.featureInputs[i].getDimensions(),
                                                                  fullyConnected.numOutputFeatures,
                                                                  fullyConnected.preserveInputPrefixDimensions));
        fullyConnected.featureOutputs.push_back(out);

        fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = out;
        fullyConnected.inputTensorFromOutputTensor[out] = fullyConnected.featureInputs[i];
        if (!fullyConnected.raggedFeatureInputs.empty()) {
            fullyConnected.raggedFeatureOutputs.push_back(fullyConnected.raggedFeatureInputs[i].withValues(out));
        }
    }
    if (fullyConnected.residualInput.has_value()) {
        for (uint32_t i = 0; i < fullyConnected.featureOutputs.size(); ++i) {
            const auto residual = fullyConnected.getRaggedResidualInput(i);
            if (!fullyConnected.raggedFeatureInputs.empty()) {
                THOR_THROW_IF_FALSE(residual.has_value());
                fullyConnected.outputTensorFromInputTensor[residual->getValues()] = fullyConnected.featureOutputs[i];
            } else {
                THOR_THROW_IF_FALSE(i == 0);
                fullyConnected.outputTensorFromInputTensor[fullyConnected.residualInput.value()] = fullyConnected.featureOutputs[0];
            }
        }
    }
    const uint32_t epilogueAuxCount = static_cast<uint32_t>(fullyConnected.epilogueAuxInputNames().size());
    for (uint32_t i = 0; i < fullyConnected.epilogueInputBindings.size(); ++i) {
        const auto& tensor = fullyConnected.epilogueInputBindings[i].second;
        const uint32_t application = fullyConnected.raggedFeatureInputs.empty() ? 0 : i / epilogueAuxCount;
        THOR_THROW_IF_FALSE(tensor.getDataType() == fullyConnected.outputDataType);
        THOR_THROW_IF_FALSE(tensor.getDimensions() == fullyConnected.featureOutputs[application].getDimensions());
        fullyConnected.outputTensorFromInputTensor[tensor] = fullyConnected.featureOutputs[application];
    }

    fullyConnected.addToNetwork(_network.value());

    return fullyConnected;
}

void FullyConnected::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw std::invalid_argument("FullyConnected::Builder requires network().");
    }
    if (_featureInputs.empty()) {
        throw std::invalid_argument("FullyConnected::Builder requires at least one featureInput().");
    }
    if (!_numOutputFeatures.has_value()) {
        throw std::invalid_argument("FullyConnected::Builder requires numOutputFeatures().");
    }
    if (_numOutputFeatures.value() == 0) {
        throw std::invalid_argument("FullyConnected numOutputFeatures must be non-zero.");
    }
    if (_numOutputFeatures.value() > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("FullyConnected numOutputFeatures exceeds the int32 cuBLASLt interface limit.");
    }
    if (_weightsInitializer == nullptr) {
        throw std::invalid_argument("FullyConnected weightsInitializer must be non-null.");
    }
    if (_hasBias.value() && _biasInitializer == nullptr) {
        throw std::invalid_argument("FullyConnected biasInitializer must be non-null when hasBias is true.");
    }
    if (!_activationExplicitlyRemoved && _activation == nullptr) {
        throw std::invalid_argument("FullyConnected activation must be non-null unless noActivation() was requested.");
    }
    if (_epilogue.has_value()) {
        FullyConnected::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
    } else if (!_epilogueInputBindings.empty()) {
        throw std::invalid_argument("FullyConnected epilogue_inputs were provided without an epilogue expression.");
    }
    const float outputDropoutProbability = _outputDropoutProbability.value_or(0.0f);
    if (!std::isfinite(outputDropoutProbability) || outputDropoutProbability < 0.0f || outputDropoutProbability >= 1.0f) {
        throw std::invalid_argument("FullyConnected outputDropoutProbability must be finite and in [0, 1).");
    }
    if (_raggedFeatureInputs.empty() && _epilogue.has_value() &&
        (_residualInput.has_value() || outputDropoutProbability > 0.0f)) {
        // Dense composition has not been independently accepted; FC-2B extends
        // only the rank-one ragged path, with an explicit epilogue-before-dropout order.
        throw std::invalid_argument(
            "FullyConnected dense custom epilogue with residual/output dropout is not yet supported.");
    }
    if (outputDropoutProbability > 0.0f &&
        _outputDataType.value() != DataType::FP16 && _outputDataType.value() != DataType::BF16 &&
        _outputDataType.value() != DataType::FP32) {
        throw std::invalid_argument("FullyConnected output dropout currently supports FP16, BF16, and FP32 output storage.");
    }
    if (_residualInput.has_value() && _raggedFeatureInputs.empty() && _featureInputs.size() != 1) {
        throw std::invalid_argument("FullyConnected dense residualInput requires exactly one featureInput.");
    }

    if (!_raggedFeatureInputs.empty()) {
        if (_raggedFeatureInputs.size() != _featureInputs.size()) {
            throw std::invalid_argument("FullyConnected cannot mix dense and ragged feature inputs.");
        }
        if (!_preserveInputPrefixDimensions.value()) {
            throw std::invalid_argument("FullyConnected(RaggedTensor) requires preserveInputPrefixDimensions=true for token-wise packed execution.");
        }
        // Multiple applications share weights but own independent logical
        // partitions. The supported fusion combinations are semantic/API
        // restrictions, not consequences of how a model was once placed.
        const bool hasRaggedEpilogue = _epilogue.has_value();
        // Activation -> output dropout -> residual is already implemented by
        // buildOutputBranch(), including its no-dropout evaluation variant.
        // An epilogue may bind zero or many same-partition auxiliary tensors.
        // Activation, dropout, and an epilogue expression are shared, while
        // each application has its own independent set of auxiliary value edges.
        if (!hasRaggedEpilogue && !_epilogueInputBindings.empty()) {
            throw std::invalid_argument("FullyConnected epilogue inputs require an epilogue expression.");
        }
        const std::vector<std::string> names = epilogueAuxInputNames();
        const uint32_t auxiliaryCount = static_cast<uint32_t>(names.size());
        if (auxiliaryCount != 0 &&
            _epilogueInputBindings.size() != _raggedFeatureInputs.size() * auxiliaryCount) {
            throw std::invalid_argument("FullyConnected ragged epilogue requires one binding of every auxiliary name per application.");
        }
        for (uint32_t i = 0; i < _epilogueInputBindings.size(); ++i) {
            if (_epilogueInputBindings[i].first != names[i % auxiliaryCount]) {
                throw std::invalid_argument("FullyConnected ragged epilogue bindings must be application-major with identical named ports.");
            }
        }
        if (_epilogueInputBindings.size() != _raggedEpilogueInputBindings.size()) {
            throw std::invalid_argument(
                "FullyConnected(RaggedTensor) epilogue auxiliary inputs must be RaggedTensor values with the exact feature row partition.");
        }
        for (uint32_t i = 0; i < _raggedFeatureInputs.size(); ++i) {
            const RaggedTensor& ragged = _raggedFeatureInputs[i];
            if (ragged.getValues().getDimensions().size() != 2 || ragged.getTrailingDimensions().size() != 1) {
                throw std::invalid_argument("FullyConnected(RaggedTensor) requires packed values shaped [capacity, features].");
            }
            if (ragged.getValues() != _featureInputs[i]) {
                throw std::invalid_argument("FullyConnected ragged feature input values do not match the primary feature input tensor.");
            }
        }
        if (_residualInput.has_value() != _raggedResidualInput.has_value()) {
            throw std::invalid_argument("FullyConnected ragged feature input requires residualInput to also be ragged.");
        }
        if (_raggedResidualInput.has_value()) {
            if (1 + _additionalRaggedResidualInputs.size() != _raggedFeatureInputs.size()) {
                throw std::invalid_argument("FullyConnected ragged residualInput requires one residual per application.");
            }
            std::set<uint64_t> residualValueIds;
            for (uint32_t i = 0; i < _raggedFeatureInputs.size(); ++i) {
                const RaggedTensor& feature = _raggedFeatureInputs[i];
                const RaggedTensor& residual = i == 0 ? _raggedResidualInput.value()
                                                             : _additionalRaggedResidualInputs[i - 1];
                if (!residual.sharesPartitionWith(feature) ||
                    residual.getMaxValuesPerRow() != feature.getMaxValuesPerRow() ||
                    residual.getValues().getDimensions() !=
                        std::vector<uint64_t>{feature.getMaxValuesPerRow(), _numOutputFeatures.value()} ||
                    !residualValueIds.insert(residual.getValues().getOriginalId()).second) {
                    throw std::invalid_argument("FullyConnected ragged residualInput must match its application's exact row partition and have a unique values identity.");
                }
            }
        } else if (!_additionalRaggedResidualInputs.empty()) {
            throw std::invalid_argument("FullyConnected ragged residuals require an application-zero residual.");
        }
        for (uint32_t i = 0; i < _raggedEpilogueInputBindings.size(); ++i) {
            const auto& [name, auxiliary] = _raggedEpilogueInputBindings[i];
            const uint32_t application = i / auxiliaryCount;
            const RaggedTensor& feature = _raggedFeatureInputs[application];
            if (!auxiliary.sharesPartitionWith(feature) ||
                auxiliary.getMaxValuesPerRow() != feature.getMaxValuesPerRow()) {
                throw std::invalid_argument("FullyConnected ragged epilogue input '" + name +
                                            "' must preserve its application's exact feature-input row partition.");
            }
        }
        if (!supportedRaggedFcStorageType(_featureInputs.front().getDataType()) ||
            !supportedRaggedFcStorageType(_weightsDataType.value()) ||
            !supportedRaggedFcStorageType(_outputDataType.value())) {
            throw std::invalid_argument("FullyConnected(RaggedTensor) currently supports fp16, bf16, and fp32 storage types.");
        }
    }

    if (_raggedFeatureInputs.empty() && (_raggedResidualInput.has_value() || !_additionalRaggedResidualInputs.empty())) {
        throw std::invalid_argument("FullyConnected dense feature input cannot use a ragged residualInput.");
    }
    if (_raggedFeatureInputs.empty() && !_raggedEpilogueInputBindings.empty()) {
        throw std::invalid_argument("FullyConnected dense feature input cannot use ragged epilogue auxiliary inputs.");
    }

    const DataType inputDataType = _featureInputs.front().getDataType();
    const std::vector<uint64_t> inputDimensions = _featureInputs.front().getDimensions();
    FullyConnected::checkedInputFeatureCount(inputDimensions, _preserveInputPrefixDimensions.value(), "feature input");
    FullyConnected::verifyFullyConnectedDataType(inputDataType, "feature input data type");
    FullyConnected::verifyFullyConnectedDataType(_weightsDataType.value(), "weightsDataType");
    FullyConnected::verifyFullyConnectedComputeDataType(_computeDataType.value());
    FullyConnected::verifyFullyConnectedDataType(_outputDataType.value(), "outputDataType");

    // Validate the matmul data-type plan against the same cuBLASLt support table used by CublasMatrixMultiply.
    (void)cublasLtMatmulDataTypesForFullyConnected(inputDataType, _weightsDataType.value(), _computeDataType.value(), _outputDataType.value());

    for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
        const Tensor& featureInput = _featureInputs[i];
        if (!featureInput.isInitialized()) {
            throw std::invalid_argument("FullyConnected featureInput " + std::to_string(i) + " is not initialized.");
        }
        if (featureInput.getDataType() != inputDataType) {
            throw std::invalid_argument("FullyConnected all feature inputs must have the same data type.");
        }
        if (_raggedFeatureInputs.empty() && featureInput.getDimensions() != inputDimensions) {
            throw std::invalid_argument("FullyConnected all dense feature inputs must have the same dimensions.");
        }
        if (!_raggedFeatureInputs.empty() &&
            (featureInput.getDimensions().size() != 2 ||
             featureInput.getDimensions().back() != inputDimensions.back())) {
            throw std::invalid_argument("FullyConnected ragged applications must share the input feature width.");
        }
        FullyConnected::checkedInputFeatureCount(
            featureInput.getDimensions(), _preserveInputPrefixDimensions.value(), "feature input " + std::to_string(i));
    }
    if (_residualInput.has_value()) {
        for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
            const Tensor& residual = _raggedFeatureInputs.empty() ? _residualInput.value()
                : (i == 0 ? _raggedResidualInput->getValues() : _additionalRaggedResidualInputs[i - 1].getValues());
            const std::vector<uint64_t> expected = FullyConnected::fullyConnectedOutputDimensions(
                _featureInputs[i].getDimensions(), _numOutputFeatures.value(), _preserveInputPrefixDimensions.value());
            if (!residual.isInitialized() || residual.getDataType() != _outputDataType.value() ||
                residual.getDimensions() != expected) {
                throw std::invalid_argument("FullyConnected residualInput must match its application's output dtype and shape.");
            }
        }
    }
    const uint32_t auxiliaryCount = static_cast<uint32_t>(epilogueAuxInputNames().size());
    for (uint32_t i = 0; i < _epilogueInputBindings.size(); ++i) {
        const auto& [name, tensor] = _epilogueInputBindings[i];
        FullyConnected::validateEpilogueAuxInputName(name);
        if (!tensor.isInitialized()) {
            throw std::invalid_argument("FullyConnected epilogue input '" + name + "' is not initialized.");
        }
        if (tensor.getDataType() != _outputDataType.value()) {
            throw std::invalid_argument("FullyConnected epilogue input '" + name + "' dtype must match outputDataType.");
        }
        const uint32_t application = _raggedFeatureInputs.empty() ? 0 : i / auxiliaryCount;
        const std::vector<uint64_t> expected = FullyConnected::fullyConnectedOutputDimensions(
            _featureInputs[application].getDimensions(), _numOutputFeatures.value(), _preserveInputPrefixDimensions.value());
        if (tensor.getDimensions() != expected) {
            throw std::invalid_argument("FullyConnected epilogue input '" + name + "' shape must match its application's output shape.");
        }
    }
}

std::shared_ptr<ThorImplementation::Layer> FullyConnected::stamp(ThorImplementation::TensorPlacement placement,
                                                                 std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                                 std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                                 Thor::Tensor connectingApiTensor,
                                                                 const bool inferenceOnly) const {
    (void)drivingApiLayer;

    THOR_THROW_IF_FALSE(initialized);
    if (!raggedFeatureInputs.empty()) {
        THOR_THROW_IF_FALSE(!inputPortIndicesForTensor(connectingApiTensor).empty());
    } else {
        THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());
    }

    std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    // Note: Network notices when a layer has already been stamped and only adds a connection; it does not re-stamp the layer.
    if (!raggedFeatureInputs.empty()) {
        // Placement may be initiated by *any* application or by a residual/epilogue
        // edge, not necessarily the first feature values. Resolve B against that
        // edge's own logical descriptor before constructing per-application geometry.
        const RaggedTensor* drivingRagged = nullptr;
        bool drivingPartitionPort = false;
        auto consider = [&](const RaggedTensor& candidate) {
            if (connectingApiTensor.getOriginalId() == candidate.getValues().getOriginalId()) {
                drivingRagged = &candidate;
                drivingPartitionPort = false;
            } else if (connectingApiTensor.getOriginalId() == candidate.getOffsets().getOriginalId() &&
                       drivingRagged == nullptr) {
                drivingRagged = &candidate;
                drivingPartitionPort = true;
            }
        };
        for (const RaggedTensor& candidate : raggedFeatureInputs) consider(candidate);
        if (raggedResidualInput.has_value()) consider(raggedResidualInput.value());
        for (const RaggedTensor& residual : additionalRaggedResidualInputs) consider(residual);
        for (const auto& [name, auxiliary] : raggedEpilogueInputBindings) {
            (void)name;
            consider(auxiliary);
        }
        if (drivingRagged == nullptr) {
            throw std::runtime_error("FullyConnected cannot resolve the logical ragged stamp driver.");
        }
        const ThorImplementation::RaggedTensorDescriptor drivingPhysical =
            SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
                *drivingRagged, drivingLayer, "FullyConnected", drivingPartitionPort);
        const uint64_t b = drivingPhysical.getBatchSize();
        if (b == 0 || b > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("FullyConnected placement has an invalid batch size.");
        }
        std::vector<uint64_t> applicationMaxValuesPerRow;
        applicationMaxValuesPerRow.reserve(raggedFeatureInputs.size());
        for (const RaggedTensor& feature : raggedFeatureInputs) {
            const auto descriptor = feature.getDescriptorForPlacement(b);
            if (descriptor.getValuesDescriptor().getDimensions().size() != 2 ||
                descriptor.getValuesDescriptor().getDataType() != feature.getValuesDataType()) {
                throw std::runtime_error("FullyConnected placement has invalid application values geometry.");
            }
            applicationMaxValuesPerRow.push_back(feature.getMaxValuesPerRow());
        }
        const auto firstPhysical = raggedFeatureInputs.front().getDescriptorForPlacement(b);
        auto physicalFullyConnected = std::make_shared<ThorImplementation::RaggedFullyConnected>(
            buildFullyConnectedExpression(
                getId(),
                hasBias,
                true,
                placement,
                weightsDataType,
                computeDataType,
                outputDataType,
                activation,
                firstPhysical.getMaxTotalValues(),
                applicationMaxValuesPerRow,
                std::string(ThorImplementation::RaggedFullyConnected::ROW_PARTITION_INPUT_NAME),
                firstPhysical.getRowPartition(),
                epilogue,
                epilogueAuxInputNames(),
                outputDropoutProbability,
                outputDropoutSeed,
                residualInput.has_value()),
            placement,
            physicalParameters,
            inferenceOnly,
            getId(),
            residualInput.has_value(),
            epilogueAuxInputNames(),
            outputDropoutProbability > 0.0f
                ? std::optional<ThorImplementation::DynamicExpressionVariantId>(FC_EVALUATION_VARIANT)
                : std::nullopt,
            isTrainingDropoutEnabled());
        physicalFullyConnected->setLayerName(getLayerType() + "#" + std::to_string(getId()));
        return physicalFullyConnected;
    }

    std::shared_ptr<ThorImplementation::CustomLayer> physicalFullyConnected = std::make_shared<ThorImplementation::FullyConnected>(
        buildFullyConnectedExpression(
            getId(),
            hasBias,
            preserveInputPrefixDimensions,
            placement,
            weightsDataType,
            computeDataType,
            outputDataType,
            activation,
            std::nullopt,
            {},
            std::nullopt,
            std::nullopt,
            epilogue,
            epilogueAuxInputNames(),
            outputDropoutProbability,
            outputDropoutSeed,
            residualInput.has_value()),
        [&]() {
            std::vector<std::string> inputNames = {"feature_input"};
            if (residualInput.has_value()) inputNames.push_back(FC_RESIDUAL_INPUT_NAME);
            std::vector<std::string> auxNames = epilogueAuxInputNames();
            inputNames.insert(inputNames.end(), auxNames.begin(), auxNames.end());
            return inputNames;
        }(),
        placement,
        physicalParameters,
        inferenceOnly,
        getId(),
        preserveInputPrefixDimensions,
        outputDropoutProbability > 0.0f
            ? std::optional<ThorImplementation::DynamicExpressionVariantId>(FC_EVALUATION_VARIANT)
            : std::nullopt,
        isTrainingDropoutEnabled());
    physicalFullyConnected->setLayerName(getLayerType() + "#" + std::to_string(getId()));

    return physicalFullyConnected;
}

json FullyConnected::architectureJson() const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    // Preserve every established canonical schema version. The version
    // distinguishes actual projection features, never model geometry modes.
    j["version"] = !raggedFeatureInputs.empty()
        ? (epilogue.has_value() ? "1.7.0" : (residualInput.has_value() ? "1.5.0" :
           (outputDropoutProbability > 0.0f ? "1.6.0" :
           (activation != nullptr ? "1.4.0" : "1.3.0"))))
        : ((residualInput.has_value() || outputDropoutProbability > 0.0f) ? "1.1.0" : getLayerVersion());
    j["layer_type"] = "fully_connected";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["num_output_features"] = numOutputFeatures;
    j["has_bias"] = hasBias;
    j["preserve_input_prefix_dimensions"] = preserveInputPrefixDimensions;
    j["weights_data_type"] = weightsDataType;
    j["compute_data_type"] = computeDataType;
    j["output_data_type"] = outputDataType;
    j["output_dropout_probability"] = outputDropoutProbability;
    j["output_dropout_seed"] = outputDropoutSeed;
    j["use_residual"] = residualInput.has_value();
    if (residualInput.has_value()) {
        if (additionalRaggedResidualInputs.empty()) {
            // Preserve the existing object grammar for every historical
            // single-application archive.
            j["residual_input"] = residualInput->architectureJson();
            if (raggedResidualInput.has_value())
                j["ragged_residual_input"] =
                    SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedResidualInput.value());
        } else {
            json residualValues = json::array();
            json residualRagged = json::array();
            for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
                const RaggedTensor residual = getRaggedResidualInput(i).value();
                residualValues.push_back(residual.getValues().architectureJson());
                residualRagged.push_back(
                    SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(residual));
            }
            j["residual_input"] = std::move(residualValues);
            j["ragged_residual_input"] = std::move(residualRagged);
        }
    }
    j["use_ragged"] = !raggedFeatureInputs.empty();
    if (!raggedFeatureInputs.empty()) {
        json raggedInputsJson = json::array();
        json raggedOutputsJson = json::array();
        for (const RaggedTensor& input : raggedFeatureInputs)
            raggedInputsJson.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(input));
        for (const RaggedTensor& output : raggedFeatureOutputs)
            raggedOutputsJson.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(output));
        j["ragged_inputs"] = std::move(raggedInputsJson);
        j["ragged_outputs"] = std::move(raggedOutputsJson);
    }

    if (activation != nullptr) {
        j["activation"] = activation->architectureJson();
    } else {
        j["activation"] = nullptr;
    }
    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value())
            serializableEpilogue = makeEpilogueDefinition(epilogue.value(), epilogueAuxInputNames());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    // Input connections
    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        inputs.push_back(featureInputs[i].architectureJson());
    }
    j["inputs"] = inputs;

    json epilogueInputs = json::array();
    for (uint32_t i = 0; i < epilogueInputBindings.size(); ++i) {
        const auto& [name, tensor] = epilogueInputBindings[i];
        json binding{{"name", name}, {"tensor", tensor.architectureJson()}};
        if (!raggedEpilogueInputBindings.empty()) {
            THOR_THROW_IF_FALSE(i < raggedEpilogueInputBindings.size());
            THOR_THROW_IF_FALSE(raggedEpilogueInputBindings[i].first == name);
            binding["ragged_tensor"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(
                raggedEpilogueInputBindings[i].second);
        }
        epilogueInputs.push_back(std::move(binding));
    }
    j["epilogue_inputs"] = epilogueInputs;

    // Output connections
    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        outputs.push_back(featureOutputs[i].architectureJson());
    }
    j["outputs"] = outputs;

    j["parameters"] = getParametersArchitectureJson()["parameters"];

    return j;
}

json FullyConnected::serialize(thor_file::TarWriter& archiveWriter,
                               Stream stream,
                               bool saveOptimizerState,
                               ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void FullyConnected::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    const std::string serializedVersion = j.at("version").get<std::string>();
    if (serializedVersion != "1.0.0" && serializedVersion != "1.1.0" &&
        serializedVersion != "1.2.0" && serializedVersion != "1.3.0" &&
        serializedVersion != "1.4.0" && serializedVersion != "1.5.0" &&
        serializedVersion != "1.6.0" && serializedVersion != "1.7.0")
        throw runtime_error("Unsupported version in FullyConnected::deserialize: " + serializedVersion);
    if (j.at("layer_type").get<std::string>() != "fully_connected")
        throw runtime_error("Layer type mismatch in FullyConnected::deserialize: " + j.at("layer_type").get<std::string>());

    // 1.2.0 was the historical packed ragged-epilogue grammar. A canonical
    // 1.7.0 reference must not become a 1.2.0 artifact merely by changing the
    // layer version string; genuinely archived 1.2.0 packed references remain
    // readable through reconstructInput().
    if (serializedVersion == "1.2.0" && j.value("use_ragged", false)) {
        for (const char* key : {"ragged_inputs", "ragged_outputs"}) {
            if (!j.contains(key) || !j.at(key).is_array()) {
                throw runtime_error("FullyConnected historical 1.2.0 requires packed ragged references.");
            }
            for (const json& reference : j.at(key)) {
                if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(reference, "FullyConnected") !=
                    SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::HISTORICAL_PACKED) {
                    throw runtime_error("FullyConnected historical 1.2.0 cannot use canonical ragged references.");
                }
            }
        }
    }
    const bool neutral = serializedVersion == "1.3.0" || serializedVersion == "1.4.0" ||
                         serializedVersion == "1.5.0" || serializedVersion == "1.6.0" ||
                         serializedVersion == "1.7.0";
    const bool neutralActivation = serializedVersion == "1.4.0";
    const bool neutralResidual = serializedVersion == "1.5.0";
    const bool neutralDropout = serializedVersion == "1.6.0";
    const bool neutralEpilogue = serializedVersion == "1.7.0";
    // The established 1.7.0 grammar already has optional activation, dropout,
    // and residual fields. A residual contributes exactly two reference objects.
    const bool canonicalResidual = neutralResidual || (neutralEpilogue && j.value("use_residual", false));
    if (neutral && (!j.is_object() || j.size() != (canonicalResidual ? 24u : 22u) ||
                    !j.at("use_ragged").get<bool>() ||
                    !j.at("preserve_input_prefix_dimensions").get<bool>() ||
                    j.at("use_residual").get<bool>() != canonicalResidual ||
                    (canonicalResidual ? (j.at("inputs").size() == 1
                        ? (!j.at("residual_input").is_object() ||
                           !j.at("ragged_residual_input").is_object())
                        : (!(neutralResidual || neutralEpilogue) || !j.at("residual_input").is_array() ||
                           !j.at("ragged_residual_input").is_array() ||
                           j.at("residual_input").size() != j.at("inputs").size() ||
                           j.at("ragged_residual_input").size() != j.at("inputs").size()))
                                     : (j.contains("residual_input") || j.contains("ragged_residual_input"))) ||
                    ((neutralDropout || neutralResidual || neutralEpilogue)
                        ? (!std::isfinite(j.at("output_dropout_probability").get<float>()) ||
                           (neutralDropout ? j.at("output_dropout_probability").get<float>() <= 0.0f
                                           : j.at("output_dropout_probability").get<float>() < 0.0f) ||
                           j.at("output_dropout_probability").get<float>() >= 1.0f)
                        : j.at("output_dropout_probability").get<float>() != 0.0f) ||
                    !j.at("output_dropout_seed").is_number_integer() ||
                    (neutralActivation ? !j.at("activation").is_object() :
                     (neutralResidual || neutralDropout || neutralEpilogue)
                         ? (!j.at("activation").is_null() && !j.at("activation").is_object())
                         : !j.at("activation").is_null()) ||
                    (neutralEpilogue ? (!j.at("epilogue").is_object() ||
                        !j.at("epilogue_inputs").is_array())
                        : (!j.at("epilogue").is_null() || !j.at("epilogue_inputs").is_array() ||
                           !j.at("epilogue_inputs").empty())) ||
                    !j.at("inputs").is_array() || j.at("inputs").empty() ||
                    !j.at("outputs").is_array() ||
                    j.at("outputs").size() != j.at("inputs").size() ||
                    !j.at("ragged_inputs").is_array() ||
                    j.at("ragged_inputs").size() != j.at("inputs").size() ||
                    !j.at("ragged_outputs").is_array() ||
                    j.at("ragged_outputs").size() != j.at("inputs").size())) {
        throw runtime_error("FullyConnected canonical archive requires supported 1.3.0-1.7.0 "
                            "projections with exact application-specific residual bindings.");
    }

    if (canonicalResidual && j.at("inputs").size() > 1) {
        std::set<uint64_t> ids;
        for (uint32_t i = 0; i < j.at("inputs").size(); ++i) {
            const json& value = j.at("residual_input").at(i);
            const json& ragged = j.at("ragged_residual_input").at(i);
            if (!value.is_object() || !ragged.is_object() ||
                !value.contains("id") || !value.at("id").is_number_unsigned() ||
                !ids.insert(value.at("id").get<uint64_t>()).second)
                throw runtime_error("FullyConnected multi-application residual references must be unique objects.");
        }
    }
    if (neutralEpilogue) {
        const uint32_t applications = static_cast<uint32_t>(j.at("inputs").size());
        const json& bindings = j.at("epilogue_inputs");
        if (bindings.size() % applications != 0) {
            throw runtime_error("FullyConnected 1.7.0 epilogue auxiliary bindings have incomplete application groups.");
        }
        const uint32_t auxiliaryCount = static_cast<uint32_t>(bindings.size() / applications);
        std::set<std::string> names;
        for (uint32_t i = 0; i < bindings.size(); ++i) {
            const json& binding = bindings.at(i);
            if (!binding.is_object() || binding.size() != 3 ||
                !binding.contains("name") || !binding.at("name").is_string() ||
                !binding.contains("tensor") || !binding.at("tensor").is_object() ||
                !binding.contains("ragged_tensor") || !binding.at("ragged_tensor").is_object() ||
                (i < auxiliaryCount && !names.insert(binding.at("name").get<std::string>()).second) ||
                (i >= auxiliaryCount && binding.at("name") != bindings.at(i % auxiliaryCount).at("name"))) {
                throw runtime_error("FullyConnected 1.7.0 epilogue auxiliary binding schema is invalid.");
            }
        }
    }
    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    std::vector<std::pair<std::string, RaggedTensor>> raggedEpilogueInputBindings;
    if (j.contains("epilogue_inputs")) {
        for (const json& epilogueInputJson : j.at("epilogue_inputs")) {
            std::string inputName = epilogueInputJson.at("name").get<std::string>();
            validateEpilogueAuxInputName(inputName);
            uint64_t originalTensorId = epilogueInputJson.at("tensor").at("id").get<uint64_t>();
            Tensor values = network->getApiTensorByOriginalId(originalTensorId);
            if (neutralEpilogue) validateNeutralFcTensorReference(epilogueInputJson.at("tensor"), values);
            epilogueInputBindings.emplace_back(inputName, values);
            if (epilogueInputJson.contains("ragged_tensor")) {
                const json& raggedJson = epilogueInputJson.at("ragged_tensor");
                if (raggedJson.at("values").at("id").get<uint64_t>() != originalTensorId) {
                    throw runtime_error("FullyConnected serialized ragged epilogue input values metadata is inconsistent.");
                }
                RaggedTensor ragged = SegmentedPrimitiveDetail::reconstructInput(
                    raggedJson, network, "FullyConnected epilogue auxiliary");
                if (neutralEpilogue)
                    validateNeutralFcRaggedReference(raggedJson, ragged);
                if (ragged.getValues().getOriginalId() != values.getOriginalId()) {
                    throw runtime_error("FullyConnected epilogue auxiliary values identity does not match the graph.");
                }
                raggedEpilogueInputBindings.emplace_back(inputName, ragged);
            }
        }
    }
    std::vector<std::string> auxInputNames;
    auxInputNames.reserve(epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)tensor;
        if (std::find(auxInputNames.begin(), auxInputNames.end(), name) == auxInputNames.end())
            auxInputNames.push_back(name);
    }

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition, auxInputNames);
    } else if (!epilogueInputBindings.empty()) {
        throw runtime_error("FullyConnected serialized epilogue_inputs require a non-null epilogue expression.");
    }

    const float outputDropoutProbability = j.value("output_dropout_probability", 0.0f);
    const int64_t outputDropoutSeed = j.value("output_dropout_seed", int64_t{0});
    std::optional<Tensor> residualInput = std::nullopt;
    std::vector<Tensor> applicationResidualValues;
    if (j.value("use_residual", false)) {
        if (!j.contains("residual_input"))
            throw runtime_error("FullyConnected serialized residual is missing residual_input metadata.");
        const json& references = j.at("residual_input");
        if (references.is_array()) {
            if (!canonicalResidual || references.size() != j.at("inputs").size() || references.size() < 2)
                throw runtime_error("FullyConnected canonical multi-application residual bindings require exact arity.");
            for (const json& reference : references) {
                const uint64_t id = reference.at("id").get<uint64_t>();
                Tensor value = network->getApiTensorByOriginalId(id);
                validateNeutralFcTensorReference(reference, value);
                applicationResidualValues.push_back(value);
            }
        } else {
            const uint64_t id = references.at("id").get<uint64_t>();
            Tensor value = network->getApiTensorByOriginalId(id);
            if (neutral) validateNeutralFcTensorReference(references, value);
            applicationResidualValues.push_back(value);
        }
        residualInput = applicationResidualValues.front();
    }
    if (serializedVersion == "1.0.0" && (residualInput.has_value() || outputDropoutProbability > 0.0f)) {
        throw runtime_error("FullyConnected 1.0.0 archives cannot contain residual/output-dropout semantics.");
    }
    if (!std::isfinite(outputDropoutProbability) || outputDropoutProbability < 0.0f || outputDropoutProbability >= 1.0f) {
        throw runtime_error("FullyConnected serialized output_dropout_probability must be finite and in [0, 1).");
    }
    if (epilogue.has_value() && (residualInput.has_value() || outputDropoutProbability > 0.0f) &&
        !j.value("use_ragged", false)) {
        throw runtime_error("FullyConnected serialized dense epilogue cannot be combined with residual/output dropout.");
    }
    if (!raggedEpilogueInputBindings.empty() && serializedVersion != "1.2.0" && !neutralEpilogue) {
        throw runtime_error("FullyConnected serialized ragged epilogue inputs require fixed 1.2.0 or neutral 1.7.0.");
    }

    FullyConnected fullyConnected(epilogue,
                                  epilogueInputBindings,
                                  raggedEpilogueInputBindings,
                                  outputDropoutProbability,
                                  outputDropoutSeed,
                                  residualInput,
                                  std::nullopt);
    fullyConnected.numOutputFeatures = j.at("num_output_features").get<uint32_t>();
    fullyConnected.hasBias = j.at("has_bias").get<bool>();
    fullyConnected.preserveInputPrefixDimensions = j.value("preserve_input_prefix_dimensions", false);
    fullyConnected.weightsDataType = j.at("weights_data_type").get<DataType>();
    fullyConnected.computeDataType = j.at("compute_data_type").get<DataType>();
    fullyConnected.outputDataType = j.at("output_data_type").get<DataType>();
    if (outputDropoutProbability > 0.0f &&
        fullyConnected.outputDataType != DataType::FP16 && fullyConnected.outputDataType != DataType::BF16 &&
        fullyConnected.outputDataType != DataType::FP32) {
        throw runtime_error("FullyConnected serialized output dropout supports FP16, BF16, and FP32 output storage.");
    }

    if (j.contains("activation") && !j.at("activation").is_null()) {
        fullyConnected.activation = Activation::deserializeTemplate(j.at("activation"));
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        fullyConnected.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }
    for (size_t i = 0; i < j.at("outputs").size(); ++i) {
        const json& outputJson = j.at("outputs").at(i);
        if (j.value("use_ragged", false)) {
            const RaggedTensor logicalInput = SegmentedPrimitiveDetail::reconstructInput(
                j.at("ragged_inputs").at(i), network, "FullyConnected");
            fullyConnected.featureOutputs.push_back(SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
                outputJson, j.at("ragged_outputs").at(i), logicalInput, "FullyConnected"));
        } else {
            fullyConnected.featureOutputs.push_back(Tensor::deserialize(outputJson, archiveReader.get()));
        }
    }
    if (fullyConnected.featureInputs.size() != fullyConnected.featureOutputs.size()) {
        throw runtime_error("FullyConnected deserialize expected equal numbers of inputs and outputs.");
    }
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged && epilogue.has_value() && serializedVersion != "1.2.0" && !neutralEpilogue) {
        throw runtime_error("FullyConnected serialized ragged custom epilogue requires fixed 1.2.0 or neutral 1.7.0.");
    }
    if (useRagged) {
        if (!j.contains("ragged_inputs") || !j.contains("ragged_outputs") ||
            j.at("ragged_inputs").size() != fullyConnected.featureInputs.size() ||
            j.at("ragged_outputs").size() != fullyConnected.featureOutputs.size()) {
            throw runtime_error("FullyConnected serialized ragged metadata does not match its input/output arity.");
        }
        // Every application shares the projection shape and learned parameters,
        // while its canonical partition remains independent. Neither application
        // order nor placement B is allowed to redefine this topology.
        std::set<uint64_t> inputValueIds;
        std::set<uint64_t> outputValueIds;
        for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
            const json& raggedInputJson = j.at("ragged_inputs").at(i);
            RaggedTensor raggedInput = SegmentedPrimitiveDetail::reconstructInput(
                raggedInputJson, network, "FullyConnected");
            const RaggedTensor raggedOutput = raggedInput.withValues(fullyConnected.featureOutputs[i]);
            if (neutral) {
                validateNeutralFcTensorReference(j.at("inputs").at(i), fullyConnected.featureInputs[i]);
                validateNeutralFcTensorReference(j.at("outputs").at(i), fullyConnected.featureOutputs[i]);
                validateNeutralFcRaggedReference(raggedInputJson, raggedInput);
                validateNeutralFcRaggedReference(j.at("ragged_outputs").at(i), raggedOutput);
            }
            SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
                j.at("ragged_outputs").at(i), raggedInputJson, raggedInput, "FullyConnected");
            if (!inputValueIds.insert(raggedInput.getValues().getOriginalId()).second ||
                !outputValueIds.insert(raggedOutput.getValues().getOriginalId()).second ||
                (i > 0 && (raggedInput.getValuesDataType() !=
                               fullyConnected.raggedFeatureInputs.front().getValuesDataType() ||
                           raggedInput.getOffsetsDataType() !=
                               fullyConnected.raggedFeatureInputs.front().getOffsetsDataType() ||
                           raggedInput.getTrailingDimensions() !=
                               fullyConnected.raggedFeatureInputs.front().getTrailingDimensions())) ||
                raggedInput.getValues().getOriginalId() != fullyConnected.featureInputs[i].getOriginalId() ||
                raggedOutput.getValues().getOriginalId() == raggedInput.getValues().getOriginalId() ||
                raggedOutput.getValues().getOriginalId() == raggedInput.getRowPartitionToken().getOriginalId() ||
                raggedInput.getValues().getOriginalId() == raggedInput.getRowPartitionToken().getOriginalId() ||
                raggedInput.getTrailingDimensions().size() != 1 ||
                raggedInput.getValuesDataType() != fullyConnected.featureInputs[i].getDataType() ||
                !supportedRaggedFcStorageType(raggedInput.getValuesDataType()) ||
                !supportedRaggedFcStorageType(raggedOutput.getValuesDataType()) ||
                raggedOutput.getTrailingDimensions() != std::vector<uint64_t>{fullyConnected.numOutputFeatures} ||
                raggedOutput.getValuesDataType() != fullyConnected.outputDataType ||
                !fullyConnected.preserveInputPrefixDimensions ||
                fullyConnected.numOutputFeatures == 0 ||
                fullyConnected.numOutputFeatures > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                throw runtime_error("FullyConnected reconstructed tensors violate projection semantics.");
            }
            fullyConnected.raggedFeatureInputs.push_back(raggedInput);
            fullyConnected.raggedFeatureOutputs.push_back(raggedOutput);
        }
    }
    if (useRagged) {
        if (fullyConnected.epilogueInputBindings.size() != fullyConnected.raggedEpilogueInputBindings.size()) {
            throw runtime_error("FullyConnected serialized ragged epilogue auxiliary inputs must carry ragged metadata.");
        }
        if (!fullyConnected.raggedEpilogueInputBindings.empty()) {
            const uint32_t auxiliaryCount = static_cast<uint32_t>(auxInputNames.size());
            if (fullyConnected.raggedEpilogueInputBindings.size() !=
                auxiliaryCount * fullyConnected.raggedFeatureInputs.size()) {
                throw runtime_error("FullyConnected serialized ragged epilogue bindings do not match application arity.");
            }
            for (uint32_t i = 0; i < fullyConnected.raggedEpilogueInputBindings.size(); ++i) {
                const auto& [name, auxiliary] = fullyConnected.raggedEpilogueInputBindings[i];
                const uint32_t application = i / auxiliaryCount;
                const RaggedTensor& feature = fullyConnected.raggedFeatureInputs[application];
                const bool sameMaxValuesPerRow =
                    auxiliary.hasMaxValuesPerRow() == feature.hasMaxValuesPerRow() &&
                    (!feature.hasMaxValuesPerRow() || auxiliary.getMaxValuesPerRow() == feature.getMaxValuesPerRow());
                if (!auxiliary.sharesPartitionWith(feature) || !sameMaxValuesPerRow ||
                    auxiliary.getValues().getDimensions() !=
                        std::vector<uint64_t>{feature.getMaxValuesPerRow(), fullyConnected.numOutputFeatures} ||
                    auxiliary.getValuesDataType() != fullyConnected.outputDataType ||
                    auxiliary.getValues().getOriginalId() ==
                        fullyConnected.featureOutputs[application].getOriginalId()) {
                    throw runtime_error("FullyConnected serialized ragged epilogue input '" + name +
                                        "' does not preserve the feature row partition.");
                }
            }
        }
    } else if (!fullyConnected.raggedEpilogueInputBindings.empty()) {
        throw runtime_error("FullyConnected dense serialized epilogue inputs cannot contain ragged metadata.");
    }
    if (residualInput.has_value()) {
        if (applicationResidualValues.size() != fullyConnected.featureOutputs.size()) {
            throw runtime_error("FullyConnected residual application count does not match its outputs.");
        }
        const bool multipleResiduals = applicationResidualValues.size() > 1;
        if (multipleResiduals && (!useRagged || !canonicalResidual)) {
            throw runtime_error("FullyConnected multiple residual applications require canonical ragged residual semantics.");
        }
        if (useRagged) {
            if (!j.contains("ragged_residual_input"))
                throw runtime_error("FullyConnected ragged residual metadata is missing.");
            const json& records = j.at("ragged_residual_input");
            if (multipleResiduals ? (!records.is_array() || records.size() != applicationResidualValues.size())
                                  : !records.is_object()) {
                throw runtime_error("FullyConnected ragged residual metadata must match application arity.");
            }
            std::set<uint64_t> residualValueIds;
            for (uint32_t i = 0; i < applicationResidualValues.size(); ++i) {
                const json& valueRecord = multipleResiduals ? j.at("residual_input").at(i) : j.at("residual_input");
                const json& raggedRecord = multipleResiduals ? records.at(i) : records;
                const Tensor& residualValue = applicationResidualValues[i];
                const RaggedTensor& feature = fullyConnected.raggedFeatureInputs[i];
                RaggedTensor residual = SegmentedPrimitiveDetail::reconstructInput(
                    raggedRecord, network, "FullyConnected residual application");
                if (neutral) {
                    validateNeutralFcTensorReference(valueRecord, residualValue);
                    validateNeutralFcRaggedReference(raggedRecord, residual);
                }
                if (residual.getValues().getOriginalId() != residualValue.getOriginalId() ||
                    residual.getValues().getDataType() != fullyConnected.outputDataType ||
                    residual.getValues().getDimensions() != fullyConnected.featureOutputs[i].getDimensions() ||
                    residual.getMaxValuesPerRow() != feature.getMaxValuesPerRow() ||
                    residual.getValues().getOriginalId() == fullyConnected.featureOutputs[i].getOriginalId() ||
                    !residual.sharesPartitionWith(feature) ||
                    !residualValueIds.insert(residualValue.getOriginalId()).second) {
                    throw runtime_error("FullyConnected residual application does not preserve its exact feature partition and output geometry.");
                }
                if (i == 0) fullyConnected.raggedResidualInput = residual;
                else fullyConnected.additionalRaggedResidualInputs.push_back(residual);
            }
        } else {
            if (multipleResiduals || j.contains("ragged_residual_input"))
                throw runtime_error("FullyConnected dense residual cannot contain multi-application ragged metadata.");
            if (residualInput->getDataType() != fullyConnected.outputDataType ||
                residualInput->getDimensions() != fullyConnected.featureOutputs.front().getDimensions())
                throw runtime_error("FullyConnected serialized dense residual descriptor is invalid.");
        }
    } else if (j.contains("ragged_residual_input")) {
        throw runtime_error("FullyConnected residual metadata is present without a residual input.");
    }

    for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
        fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = fullyConnected.featureOutputs[i];
        fullyConnected.inputTensorFromOutputTensor[fullyConnected.featureOutputs[i]] = fullyConnected.featureInputs[i];
    }
    if (fullyConnected.residualInput.has_value()) {
        for (uint32_t i = 0; i < applicationResidualValues.size(); ++i)
            fullyConnected.outputTensorFromInputTensor[applicationResidualValues[i]] = fullyConnected.featureOutputs[i];
    }
    if (!fullyConnected.epilogueInputBindings.empty()) {
        const uint32_t auxiliaryCount = static_cast<uint32_t>(auxInputNames.size());
        for (uint32_t i = 0; i < fullyConnected.epilogueInputBindings.size(); ++i) {
            const auto& [name, tensor] = fullyConnected.epilogueInputBindings[i];
            (void)name;
            const uint32_t application = useRagged ? i / auxiliaryCount : 0;
            if (tensor.getDataType() != fullyConnected.featureOutputs[application].getDataType()) {
                throw runtime_error("FullyConnected serialized epilogue input dtype does not match its application's output dtype.");
            }
            if (tensor.getDimensions() != fullyConnected.featureOutputs[application].getDimensions()) {
                throw runtime_error("FullyConnected serialized epilogue input shape does not match its application's output shape.");
            }
            fullyConnected.outputTensorFromInputTensor[tensor] = fullyConnected.featureOutputs[application];
        }
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("FullyConnected parameters must be an object keyed by parameter name.");
        }
        if (neutral) {
            // Check the learned inference-state shapes against this layer's
            // semantic projection, not against placement-selected B*M.
            if (parametersJson.size() != (fullyConnected.hasBias ? 2u : 1u)) {
                throw runtime_error("FullyConnected neutral parameter inventory differs from the projection configuration.");
            }
            const uint64_t inFeatures = fullyConnected.raggedFeatureInputs.front().getTrailingDimensions().at(0);
            auto checkParameter = [&](const char* name, const std::vector<uint64_t>& shape, DataType dtype) {
                const json& entry = parametersJson.at(name);
                if (!entry.is_object() || entry.at("name").get<std::string>() != name) {
                    throw runtime_error("FullyConnected neutral parameter identity is invalid.");
                }
                if (entry.contains("storage")) {
                    const json& storage = entry.at("storage");
                    if (storage.at("dimensions").get<std::vector<uint64_t>>() != shape ||
                        storage.at("data_type").get<DataType>() != dtype) {
                        throw runtime_error("FullyConnected neutral stored parameter shape or dtype is invalid.");
                    }
                } else if (entry.at("shape").get<std::vector<uint64_t>>() != shape ||
                           entry.at("dtype").get<DataType>() != dtype) {
                    throw runtime_error("FullyConnected neutral parameter shape or dtype is invalid.");
                }
            };
            checkParameter("weights", {inFeatures, fullyConnected.numOutputFeatures}, fullyConnected.weightsDataType);
            if (fullyConnected.hasBias)
                checkParameter("biases", {fullyConnected.numOutputFeatures}, fullyConnected.outputDataType);
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            fullyConnected.addParameter(std::make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }

    if (!fullyConnected.hasParameter("weights")) {
        throw runtime_error("FullyConnected deserialize did not find required weights parameter.");
    }
    if (fullyConnected.hasBias && !fullyConnected.hasParameter("biases")) {
        throw runtime_error("FullyConnected deserialize did not find required biases parameter.");
    }

    fullyConnected.initialized = true;
    fullyConnected.addToNetwork(network);
}


}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("fully_connected", &Thor::FullyConnected::deserialize);
    return true;
}();
}  // namespace
