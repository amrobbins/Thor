#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <optional>
#include <set>
#include <sstream>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

constexpr const char* RAGGED_ROW_PARTITION_EXPRESSION_INPUT = "__ragged_row_partition_runtime";

bool supportedRaggedFcStorageType(DataType dataType) {
    return dataType == DataType::FP16 || dataType == DataType::BF16 || dataType == DataType::FP32;
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

struct ExpressionInputDataTypes {
    std::optional<DataType> computeDataType;
    std::optional<DataType> outputDataType;
};

ExpressionInputDataTypes expressionInputDataTypes(const ThorImplementation::Expression& expression,
                                                   const std::string& inputName) {
    const ThorImplementation::PhysicalExpression physicalExpression = expression.expression();
    std::optional<ExpressionInputDataTypes> resolved;

    for (const ThorImplementation::ExprNode& node : physicalExpression.nodes) {
        if (node.op != ThorImplementation::ExprOp::INPUT) {
            continue;
        }
        if (node.input_slot >= physicalExpression.inputs.size()) {
            throw std::runtime_error("FullyConnected epilogue input node has an invalid input slot.");
        }
        if (physicalExpression.inputs[node.input_slot].name != inputName) {
            continue;
        }

        const ExpressionInputDataTypes candidate{node.compute_dtype, node.output_dtype};
        if (resolved.has_value() &&
            (resolved->computeDataType != candidate.computeDataType ||
             resolved->outputDataType != candidate.outputDataType)) {
            throw std::runtime_error("FullyConnected epilogue input '" + inputName +
                                     "' is used with inconsistent dtype annotations.");
        }
        resolved = candidate;
    }

    if (!resolved.has_value()) {
        throw std::runtime_error("FullyConnected epilogue expression does not contain expected input '" + inputName + "'.");
    }
    return resolved.value();
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
                                                                    std::optional<std::string> rowPartitionInputName,
                                                                    std::optional<ThorImplementation::Expression> epilogue,
                                                                    std::vector<std::string> epilogueAuxInputNames) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    std::vector<std::string> expectedInputNames = {"feature_input"};
    if (rowPartitionInputName.has_value()) {
        expectedInputNames.push_back(rowPartitionInputName.value());
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
         rowPartitionInputName = std::move(rowPartitionInputName),
         epilogue,
         epilogueAuxInputNames = std::move(epilogueAuxInputNames)](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            Tensor featureInputTensor = inputs.at("feature_input");
            std::optional<Tensor> rowPartitionTensor;
            if (rowPartitionInputName.has_value()) {
                rowPartitionTensor = inputs.at(rowPartitionInputName.value());
                const ThorImplementation::TensorDescriptor descriptor = rowPartitionTensor->getDescriptor();
                if (descriptor.getNumDimensions() != 1 || descriptor.getDimensions()[0] == 0 ||
                    !ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(descriptor.getDataType())) {
                    throw std::runtime_error("Ragged FullyConnected row-partition input must be a canonical offsets tensor.");
                }
                if (rowPartitionTensor->getPlacement() != placement) {
                    throw std::runtime_error("Ragged FullyConnected row-partition input placement does not match the layer placement.");
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

            auto fin = Expression::input("feature_input", featureInputTensor.getDataType(), featureInputTensor.getDataType());
            if (originalFeatureInputDimensions != logicalFeatureInputDimensions) {
                fin = fin.reshape(logicalFeatureInputDimensions);
            }
            if (rowPartitionTensor.has_value()) {
                if (!packedRowCapacity.has_value() || logicalFeatureInputDimensions[0] != packedRowCapacity.value()) {
                    throw std::runtime_error("Ragged FullyConnected packed capacity does not match its flattened values rows.");
                }
                const ThorImplementation::TensorDescriptor offsetsDescriptor = rowPartitionTensor->getDescriptor();
                const uint64_t raggedBatchSize = offsetsDescriptor.getDimensions()[0] - 1;
                Expression offsets = Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT,
                                                       offsetsDescriptor.getDataType(),
                                                       offsetsDescriptor.getDataType());
                fin = fin.withRaggedRuntimeExtent(offsets,
                                                  raggedBatchSize,
                                                  packedRowCapacity.value(),
                                                  logicalFeatureInputDimensions[1]);
            }
            auto w = Expression::input("weights", weightsDataType, weightsDataType);

            // [batch, in_features] @ [in_features, out_features]
            Expression fout = Expression::matmul(fin, w, false, false, computeDataType, outputDataType, packedRowCapacity);

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
            if (epilogue.has_value()) {
                // Keep the entire epilogue in the folded matmul geometry.  In training, applying a rank-3
                // residual epilogue after first reshaping the primary result to [B, ...prefix, O] causes the
                // same upstream gradient tensor to participate in both the public rank-3 domain and the
                // folded [B*prefix, O] domain in one backward fused equation.  The fused layout planner
                // cannot assign one broadcast layout to that input.  Flatten each public-shape auxiliary
                // epilogue input instead, apply the epilogue at [B*prefix, O], and restore the public shape
                // only once after the epilogue.
                Expression effectiveEpilogue = epilogue.value();
                if (runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                    for (const std::string& auxInputName : epilogueAuxInputNames) {
                        const ExpressionInputDataTypes inputDataTypes = expressionInputDataTypes(effectiveEpilogue, auxInputName);
                        Expression flattenedAuxInput =
                            Expression::input(auxInputName, inputDataTypes.computeDataType, inputDataTypes.outputDataType)
                                .reshape(matmulOutputDimensions);
                        effectiveEpilogue = effectiveEpilogue.substituteInput(auxInputName, flattenedAuxInput);
                    }
                }
                fout = FullyConnected::applyEpilogue(fout, effectiveEpilogue);
            }

            if (runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                fout = fout.reshape(runtimeFeatureOutputDimensions);
            }

            // The API layer's declared output tensor dtype is authoritative.
            fout = fout.withOutputDType(outputDataType);

            if (rowPartitionTensor.has_value()) {
                // MATMUL consumes the input-side extent for bucket dispatch. Re-attach
                // the same logical row partition to the public FC result so valuewise
                // epilogues and autodiff (notably db broadcast reduction) remain
                // active-aware without requiring producer-side tail canonicalization.
                if (!packedRowCapacity.has_value() || runtimeFeatureOutputDimensions.empty() ||
                    runtimeFeatureOutputDimensions[0] != packedRowCapacity.value()) {
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
                const ThorImplementation::TensorDescriptor offsetsDescriptor = rowPartitionTensor->getDescriptor();
                Expression offsets = Expression::input(RAGGED_ROW_PARTITION_EXPRESSION_INPUT,
                                                       offsetsDescriptor.getDataType(),
                                                       offsetsDescriptor.getDataType());
                fout = fout.withRaggedRuntimeExtent(offsets,
                                                    offsetsDescriptor.getDimensions()[0] - 1,
                                                    packedRowCapacity.value(),
                                                    outputElementsPerPackedValue);
            }

            auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

            DynamicExpression::TensorMap stampInputs = inputs;
            DynamicExpression::TensorMap preForwardOnlyInputs;
            std::function<void(Stream&)> preForwardHook;
            if (rowPartitionInputName.has_value()) {
                stampInputs.erase(rowPartitionInputName.value());
                stampInputs.emplace(RAGGED_ROW_PARTITION_EXPRESSION_INPUT, rowPartitionTensor.value());
                preForwardOnlyInputs.emplace(rowPartitionInputName.value(), rowPartitionTensor.value());
                // The public offsets port remains a non-differentiable structural dependency of
                // RaggedFullyConnected.  The private binding gives packed Expression stages direct
                // access to the same canonical offsets tensor for runtime bucket selection.
                preForwardHook = [](Stream&) {};
            }

            return DynamicExpressionBuild{
                .equation = std::make_shared<FusedEquation>(
                    FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
                .stamp_inputs = std::move(stampInputs),
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = std::move(preForwardHook),
                .serialized_definition = nullptr,
                .execution_variants = {},
                .evaluation_variant_id = std::nullopt,
                .pre_forward_only_inputs = std::move(preForwardOnlyInputs),
            };
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
    // Compute follows the feature-input storage type by default. In particular, FP32 inputs use strict
    // FP32 compute; callers may explicitly request TF32 Tensor Core compute with computeDataType(DataType::TF32).
    (void)weightsDataType;
    (void)outputDataType;
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
        names.push_back(name);
    }
    return names;
}

std::vector<Tensor> FullyConnected::getFeatureInputs() const {
    if (!raggedFeatureInputs.empty()) {
        std::vector<Tensor> inputs;
        inputs.reserve(raggedFeatureInputs.size() * 2);
        for (const RaggedTensor& ragged : raggedFeatureInputs) {
            inputs.push_back(ragged.getValues());
            inputs.push_back(ragged.getOffsets());
        }
        return inputs;
    }

    std::vector<Tensor> inputs = featureInputs;
    inputs.reserve(inputs.size() + epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)name;
        inputs.push_back(tensor);
    }
    return inputs;
}

std::vector<uint32_t> FullyConnected::inputPortIndicesForTensor(Tensor tensor) const {
    std::vector<uint32_t> ports;
    if (!raggedFeatureInputs.empty()) {
        for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
            if (tensor.getOriginalId() == raggedFeatureInputs[i].getValues().getOriginalId()) {
                ports.push_back(i * 2);
            }
            if (tensor.getOriginalId() == raggedFeatureInputs[i].getOffsets().getOriginalId()) {
                ports.push_back(i * 2 + 1);
            }
        }
        return ports;
    }

    if (!featureInputs.empty() && tensor.getOriginalId() == featureInputs[0].getOriginalId()) {
        ports.push_back(0);
    }
    for (uint32_t i = 0; i < epilogueInputBindings.size(); ++i) {
        if (tensor.getOriginalId() == epilogueInputBindings[i].second.getOriginalId()) {
            ports.push_back(i + 1);
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
        for (uint32_t applicationIndex = 0; applicationIndex < raggedFeatureInputs.size(); ++applicationIndex) {
            const uint32_t valuesPort = applicationIndex * 2;
            const uint32_t offsetsPort = valuesPort + 1;
            if (!connectedInputPortIndices.contains(valuesPort) || !connectedInputPortIndices.contains(offsetsPort) ||
                emittedRaggedOutputApplications.contains(applicationIndex)) {
                continue;
            }
            THOR_THROW_IF_FALSE(applicationIndex < featureOutputs.size());
            emittedRaggedOutputApplications.insert(applicationIndex);
            readyOutputs.push_back(featureOutputs[applicationIndex]);
        }
        return readyOutputs;
    }

    if (epilogueInputBindings.empty()) {
        return {getFeatureOutput(inputTensor)};
    }

    (void)getFeatureOutput(inputTensor);

    if (emittedFeatureOutputAfterAllInputsConnected) {
        return {};
    }
    const uint32_t requiredInputPorts = static_cast<uint32_t>(1 + epilogueInputBindings.size());
    if (connectedInputPortIndices.size() != requiredInputPorts) {
        return {};
    }

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

    if (epilogueInputBindings.empty()) {
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
    if (!raggedFeatureInputs.empty() || !epilogueInputBindings.empty()) {
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

    if (!_epilogueInputBindings.empty() && _featureInputs.size() != 1) {
        throw std::invalid_argument("FullyConnected epilogue auxiliary inputs currently require exactly one feature input.");
    }

    verifyConfig();

    FullyConnected fullyConnected(_epilogue, _epilogueInputBindings);

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
            fullyConnected.raggedFeatureOutputs.emplace_back(out, fullyConnected.raggedFeatureInputs[i].getOffsets());
        }
    }
    for (const auto& [name, tensor] : fullyConnected.epilogueInputBindings) {
        (void)name;
        THOR_THROW_IF_FALSE(tensor.getDataType() == fullyConnected.outputDataType);
        THOR_THROW_IF_FALSE(tensor.getDimensions() == fullyConnected.featureOutputs[0].getDimensions());
        fullyConnected.outputTensorFromInputTensor[tensor] = fullyConnected.featureOutputs[0];
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

    if (!_raggedFeatureInputs.empty()) {
        if (_raggedFeatureInputs.size() != _featureInputs.size()) {
            throw std::invalid_argument("FullyConnected cannot mix dense and ragged feature inputs.");
        }
        if (!_preserveInputPrefixDimensions.value()) {
            throw std::invalid_argument("FullyConnected(RaggedTensor) requires preserveInputPrefixDimensions=true for token-wise packed execution.");
        }
        if (_epilogue.has_value() || !_epilogueInputBindings.empty()) {
            throw std::invalid_argument("FullyConnected(RaggedTensor) does not yet support the dense FullyConnected epilogue surface.");
        }
        for (uint32_t i = 0; i < _raggedFeatureInputs.size(); ++i) {
            const RaggedTensor& ragged = _raggedFeatureInputs[i];
            if (ragged.getValues().getDimensions().size() != 2 || ragged.getTrailingDimensions().size() != 1) {
                throw std::invalid_argument("FullyConnected(RaggedTensor) currently requires packed values shaped [max_total_values, features].");
            }
            if (ragged.getValues() != _featureInputs[i]) {
                throw std::invalid_argument("FullyConnected ragged feature input values do not match the primary feature input tensor.");
            }
        }
        if (!supportedRaggedFcStorageType(_featureInputs.front().getDataType()) ||
            !supportedRaggedFcStorageType(_weightsDataType.value()) ||
            !supportedRaggedFcStorageType(_outputDataType.value())) {
            throw std::invalid_argument("FullyConnected(RaggedTensor) currently supports fp16, bf16, and fp32 storage types.");
        }
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
        if (featureInput.getDimensions() != inputDimensions) {
            throw std::invalid_argument("FullyConnected all feature inputs must have the same dimensions.");
        }
        FullyConnected::checkedInputFeatureCount(
            featureInput.getDimensions(), _preserveInputPrefixDimensions.value(), "feature input " + std::to_string(i));
    }
    const std::vector<uint64_t> expectedEpilogueInputDims =
        FullyConnected::fullyConnectedOutputDimensions(inputDimensions, _numOutputFeatures.value(), _preserveInputPrefixDimensions.value());
    for (const auto& [name, tensor] : _epilogueInputBindings) {
        FullyConnected::validateEpilogueAuxInputName(name);
        if (!tensor.isInitialized()) {
            throw std::invalid_argument("FullyConnected epilogue input '" + name + "' is not initialized.");
        }
        if (tensor.getDataType() != _outputDataType.value()) {
            throw std::invalid_argument("FullyConnected epilogue input '" + name + "' dtype must match outputDataType.");
        }
        if (tensor.getDimensions() != expectedEpilogueInputDims) {
            throw std::invalid_argument("FullyConnected epilogue input '" + name + "' shape must match feature output shape.");
        }
    }
}

std::shared_ptr<ThorImplementation::Layer> FullyConnected::stamp(ThorImplementation::TensorPlacement placement,
                                                                 std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                                 std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                                 Thor::Tensor connectingApiTensor,
                                                                 const bool inferenceOnly) const {
    (void)drivingLayer;
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
        const std::vector<uint64_t> inputDimensions = raggedFeatureInputs.front().getValues().getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() == 2);
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
                inputDimensions[0],
                std::string(ThorImplementation::RaggedFullyConnected::ROW_PARTITION_INPUT_NAME),
                std::nullopt,
                {}),
            placement,
            physicalParameters,
            inferenceOnly,
            getId());
        physicalFullyConnected->setLayerName(getLayerType() + "#" + std::to_string(getId()));
        return physicalFullyConnected;
    }

    std::shared_ptr<ThorImplementation::CustomLayer> physicalFullyConnected = std::make_shared<ThorImplementation::CustomLayer>(
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
            std::nullopt,
            epilogue,
            epilogueAuxInputNames()),
        [&]() {
            std::vector<std::string> inputNames = {"feature_input"};
            std::vector<std::string> auxNames = epilogueAuxInputNames();
            inputNames.insert(inputNames.end(), auxNames.begin(), auxNames.end());
            return inputNames;
        }(),
        std::vector<std::string>{"feature_output"},
        placement,
        physicalParameters,
        inferenceOnly,
        getId());
    physicalFullyConnected->setLayerName(getLayerType() + "#" + std::to_string(getId()));

    return physicalFullyConnected;
}

json FullyConnected::architectureJson() const {
    // Multi-layers will only serialize the single layer, itself.
    // The other layers will each serialize themselves when walking the api level layer graph that has been added to the network

    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "fully_connected";
    string layerName = string("layer") + to_string(getId());
    j["layer_name"] = layerName;
    j["num_output_features"] = numOutputFeatures;
    j["has_bias"] = hasBias;
    j["preserve_input_prefix_dimensions"] = preserveInputPrefixDimensions;
    j["weights_data_type"] = weightsDataType;
    j["compute_data_type"] = computeDataType;
    j["output_data_type"] = outputDataType;
    j["use_ragged"] = !raggedFeatureInputs.empty();
    if (!raggedFeatureInputs.empty()) {
        json raggedInputsJson = json::array();
        json raggedOutputsJson = json::array();
        for (const RaggedTensor& input : raggedFeatureInputs) raggedInputsJson.push_back(input.architectureJson());
        for (const RaggedTensor& output : raggedFeatureOutputs) raggedOutputsJson.push_back(output.architectureJson());
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
    for (const auto& [name, tensor] : epilogueInputBindings) {
        epilogueInputs.push_back(json{{"name", name}, {"tensor", tensor.architectureJson()}});
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
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in FullyConnected::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "fully_connected")
        throw runtime_error("Layer type mismatch in FullyConnected::deserialize: " + j.at("layer_type").get<std::string>());

    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    if (j.contains("epilogue_inputs")) {
        for (const json& epilogueInputJson : j.at("epilogue_inputs")) {
            std::string inputName = epilogueInputJson.at("name").get<std::string>();
            validateEpilogueAuxInputName(inputName);
            uint64_t originalTensorId = epilogueInputJson.at("tensor").at("id").get<uint64_t>();
            epilogueInputBindings.emplace_back(inputName, network->getApiTensorByOriginalId(originalTensorId));
        }
    }
    std::vector<std::string> auxInputNames;
    auxInputNames.reserve(epilogueInputBindings.size());
    for (const auto& [name, tensor] : epilogueInputBindings) {
        (void)tensor;
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

    FullyConnected fullyConnected(epilogue, epilogueInputBindings);
    fullyConnected.numOutputFeatures = j.at("num_output_features").get<uint32_t>();
    fullyConnected.hasBias = j.at("has_bias").get<bool>();
    fullyConnected.preserveInputPrefixDimensions = j.value("preserve_input_prefix_dimensions", false);
    fullyConnected.weightsDataType = j.at("weights_data_type").get<DataType>();
    fullyConnected.computeDataType = j.at("compute_data_type").get<DataType>();
    fullyConnected.outputDataType = j.at("output_data_type").get<DataType>();

    if (j.contains("activation") && !j.at("activation").is_null()) {
        fullyConnected.activation = Activation::deserializeTemplate(j.at("activation"));
    }

    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        fullyConnected.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }
    for (const json& outputJson : j.at("outputs")) {
        fullyConnected.featureOutputs.push_back(Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (fullyConnected.featureInputs.size() != fullyConnected.featureOutputs.size()) {
        throw runtime_error("FullyConnected deserialize expected equal numbers of inputs and outputs.");
    }
    const bool useRagged = j.value("use_ragged", false);
    if (useRagged) {
        if (!j.contains("ragged_inputs") || !j.contains("ragged_outputs") ||
            j.at("ragged_inputs").size() != fullyConnected.featureInputs.size() ||
            j.at("ragged_outputs").size() != fullyConnected.featureOutputs.size()) {
            throw runtime_error("FullyConnected serialized ragged metadata does not match its input/output arity.");
        }
        for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
            const json& raggedInputJson = j.at("ragged_inputs").at(i);
            const uint64_t inputOffsetsId = raggedInputJson.at("offsets").at("id").get<uint64_t>();
            RaggedTensor raggedInput(fullyConnected.featureInputs[i], network->getApiTensorByOriginalId(inputOffsetsId));
            if (raggedInput.getBatchSize() != raggedInputJson.at("batch_size").get<uint64_t>() ||
                raggedInput.getMaxTotalValues() != raggedInputJson.at("max_total_values").get<uint64_t>()) {
                throw runtime_error("FullyConnected serialized ragged input metadata does not match reconstructed tensors.");
            }
            fullyConnected.raggedFeatureInputs.push_back(raggedInput);
            fullyConnected.raggedFeatureOutputs.emplace_back(fullyConnected.featureOutputs[i], raggedInput.getOffsets());
            const json& raggedOutputJson = j.at("ragged_outputs").at(i);
            if (raggedOutputJson.at("offsets").at("id").get<uint64_t>() != inputOffsetsId) {
                throw runtime_error("FullyConnected serialized ragged output must preserve the input row partition.");
            }
        }
    }
    for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
        fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = fullyConnected.featureOutputs[i];
        fullyConnected.inputTensorFromOutputTensor[fullyConnected.featureOutputs[i]] = fullyConnected.featureInputs[i];
    }
    if (!fullyConnected.epilogueInputBindings.empty()) {
        if (fullyConnected.featureOutputs.size() != 1) {
            throw runtime_error("FullyConnected serialized epilogue_inputs require exactly one primary feature output.");
        }
        for (const auto& [name, tensor] : fullyConnected.epilogueInputBindings) {
            (void)name;
            if (tensor.getDataType() != fullyConnected.featureOutputs[0].getDataType()) {
                throw runtime_error("FullyConnected serialized epilogue input dtype does not match the feature output dtype.");
            }
            if (tensor.getDimensions() != fullyConnected.featureOutputs[0].getDimensions()) {
                throw runtime_error("FullyConnected serialized epilogue input shape does not match the feature output shape.");
            }
            fullyConnected.outputTensorFromInputTensor[tensor] = fullyConnected.featureOutputs[0];
        }
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("FullyConnected parameters must be an object keyed by parameter name.");
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
