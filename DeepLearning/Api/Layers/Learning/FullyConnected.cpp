#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <optional>
#include <set>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

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

    // This mirrors EquationCompiler's safe fallback: when cuBLASLt does not expose the requested mixed A/B plan,
    // the expression path can cast the matrix inputs into the resolved output dtype before the matmul stage.
    const MatmulDataTypes outputDataTypes{outputDataType, outputDataType, outputDataType, outputDataType, computeDataType};
    if (isSupportedCublasLtMatmulDataTypesForFullyConnected(outputDataTypes)) {
        return outputDataTypes;
    }

    throw std::invalid_argument("FullyConnected requested dtype plan is unsupported by Thor's cuBLASLt matmul path. input=" +
                                fullyConnectedDataTypeName(inputDataType) + ", weights=" + fullyConnectedDataTypeName(weightsDataType) +
                                ", compute=" + fullyConnectedDataTypeName(computeDataType) +
                                ", output=" + fullyConnectedDataTypeName(outputDataType) + ".");
}

ThorImplementation::DynamicExpression buildFullyConnectedExpression(bool hasBias,
                                                                    bool preserveInputPrefixDimensions,
                                                                    ThorImplementation::TensorPlacement placement,
                                                                    DataType weightsDataType,
                                                                    DataType computeDataType,
                                                                    DataType outputDataType,
                                                                    std::shared_ptr<Thor::Activation> activation,
                                                                    std::optional<ThorImplementation::Expression> epilogue,
                                                                    std::vector<std::string> epilogueAuxInputNames) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    std::vector<std::string> expectedInputNames = {"feature_input"};
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
        [hasBias,
         preserveInputPrefixDimensions,
         placement,
         weightsDataType,
         computeDataType,
         outputDataType,
         activation = std::move(activationClone),
         epilogue,
         epilogueAuxInputNames = std::move(epilogueAuxInputNames)](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            Tensor featureInputTensor = inputs.at("feature_input");
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
                throw std::runtime_error("FullyConnected input feature count does not match weights rows.");
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
            auto w = Expression::input("weights", weightsDataType, weightsDataType);

            // [batch, in_features] @ [in_features, out_features]
            Expression fout = Expression::matmul(fin, w, false, false, computeDataType, outputDataType);

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
            if (epilogue.has_value() && runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                // Apply epilogues in the public output shape.  This keeps tokenwise FullyConnected epilogue
                // inputs/output shaped [batch, ...prefix, out_features] instead of exposing the folded matmul batch.
                fout = fout.reshape(runtimeFeatureOutputDimensions);
            }
            for (const std::string& auxInputName : epilogueAuxInputNames) {
                const Tensor& auxTensor = inputs.at(auxInputName);
                const std::vector<uint64_t>& expectedAuxShape = epilogue.has_value() ? runtimeFeatureOutputDimensions : matmulOutputDimensions;
                if (auxTensor.getDimensions() != expectedAuxShape) {
                    throw std::runtime_error("FullyConnected epilogue auxiliary input '" + auxInputName +
                                             "' shape must match the fully connected feature output shape.");
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
                fout = FullyConnected::applyEpilogue(fout, epilogue.value());
            }

            if (!epilogue.has_value() && runtimeFeatureOutputDimensions != matmulOutputDimensions) {
                fout = fout.reshape(runtimeFeatureOutputDimensions);
            }

            // The API layer's declared output tensor dtype is authoritative.
            fout = fout.withOutputDType(outputDataType);

            auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
                inputs,
                {},
                outputs,
                {},
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
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputConnectionCursorByTensorOriginalId.clear();
}

int FullyConnected::getConnectionType(Tensor connectingTensor) const {
    if (!epilogueInputBindings.empty()) {
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
        _preserveInputPrefixDimensions = false;
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
    THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    // Note: Network notices when a layer has already been stamped and only adds a connection; it does not re-stamp the layer.
    std::shared_ptr<ThorImplementation::CustomLayer> physicalFullyConnected = std::make_shared<ThorImplementation::CustomLayer>(
        buildFullyConnectedExpression(
            hasBias,
            preserveInputPrefixDimensions,
            placement,
            weightsDataType,
            computeDataType,
            outputDataType,
            activation,
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
