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

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

bool isFullyConnectedFloatingDataType(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::FP8_E4M3:
        case Tensor::DataType::FP8_E5M2:
        case Tensor::DataType::FP16:
        case Tensor::DataType::BF16:
        case Tensor::DataType::FP32:
            return true;
        default:
            return false;
    }
}

std::string fullyConnectedDataTypeName(Tensor::DataType dataType) {
    return ThorImplementation::TensorDescriptor::getElementTypeName(dataType);
}

cudaDataType_t cublasLtCudaDataTypeForFullyConnected(Tensor::DataType dataType) {
    switch (dataType) {
        case Tensor::DataType::FP32:
            return CUDA_R_32F;
        case Tensor::DataType::BF16:
            return CUDA_R_16BF;
        case Tensor::DataType::FP16:
            return CUDA_R_16F;
        case Tensor::DataType::FP8_E4M3:
            return CUDA_R_8F_E4M3;
        case Tensor::DataType::FP8_E5M2:
            return CUDA_R_8F_E5M2;
        default:
            throw std::invalid_argument("FullyConnected cuBLASLt dtype check does not support " +
                                        fullyConnectedDataTypeName(dataType) + ".");
    }
}

std::optional<cublasComputeType_t> cublasLtComputeTypeForFullyConnected(Tensor::DataType computeDataType) {
    switch (computeDataType) {
        case Tensor::DataType::FP32:
            return CUBLAS_COMPUTE_32F;
        case Tensor::DataType::FP16:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        case Tensor::DataType::BF16:
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
    Tensor::DataType inputDataType,
    Tensor::DataType weightsDataType,
    Tensor::DataType computeDataType,
    Tensor::DataType outputDataType) {
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
                                                                    ThorImplementation::TensorPlacement placement,
                                                                    Tensor::DataType weightsDataType,
                                                                    Tensor::DataType computeDataType,
                                                                    Tensor::DataType outputDataType,
                                                                    std::shared_ptr<Thor::Activation> activation,
                                                                    std::optional<ThorImplementation::Expression> epilogue) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    std::vector<std::string> expectedInputNames = {"feature_input", "weights"};
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
        [hasBias, placement, weightsDataType, computeDataType, outputDataType, activation = std::move(activationClone), epilogue](
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

            // Treat any rank > 2 input as [batch, flattened_features] for the matrix multiply, without touching the
            // original Tensor object owned by the surrounding graph. Tensor is a lightweight metadata/storage alias,
            // so this reshape changes only this DynamicExpression's logical view.
            if (featureInputDimensions.size() > 2) {
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
                featureInputTensor.reshape({batchSize, flattenedFeatures});
                featureInputDimensions = featureInputTensor.getDimensions();
            }

            if (featureInputDimensions.size() != 2) {
                throw std::runtime_error("FullyConnected logical feature input tensor must be rank 2 after flattening.");
            }
            if (featureInputDimensions[0] == 0 || featureInputDimensions[1] == 0) {
                throw std::runtime_error("FullyConnected logical feature input tensor dimensions must be non-zero.");
            }
            if (featureInputDimensions[1] != wTensor.getDimensions()[0]) {
                throw std::runtime_error("FullyConnected input feature count does not match weights rows.");
            }
            if (outputs.contains("feature_output")) {
                const Tensor& featureOutputTensor = outputs.at("feature_output");
                if (featureOutputTensor.getDimensions().size() != 2) {
                    throw std::runtime_error("FullyConnected feature output tensor must be rank 2.");
                }
                if (featureOutputTensor.getDimensions()[0] != featureInputDimensions[0] ||
                    featureOutputTensor.getDimensions()[1] != wTensor.getDimensions()[1]) {
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
            auto w = Expression::input("weights", weightsDataType, weightsDataType);

            // [batch, in_features] @ [in_features, out_features]
            Expression fout = Expression::matmul(fin, w, false, false, computeDataType, outputDataType);

            if (hasBias) {
                const Tensor& bTensor = inputs.at("biases");
                if (bTensor.getDimensions().size() != 1 || bTensor.getDimensions()[0] != wTensor.getDimensions()[1]) {
                    throw std::runtime_error("FullyConnected biases tensor dimensions are incompatible with the weights tensor.");
                }
                if (bTensor.getDataType() != weightsDataType) {
                    throw std::runtime_error("FullyConnected biases tensor dtype does not match weightsDataType.");
                }
                if (bTensor.getPlacement() != placement) {
                    throw std::runtime_error("FullyConnected biases tensor placement does not match the layer placement.");
                }

                auto b = Expression::input("biases", weightsDataType, weightsDataType);

                // Broadcast [out_features] over batch.
                fout = fout + b;
            }

            if (activation != nullptr) {
                fout = activation->toExpression(fout);
            }
            if (epilogue.has_value()) {
                fout = FullyConnected::applyEpilogue(fout, epilogue.value());
            }

            // The API layer's declared output tensor dtype is authoritative.
            fout = fout.withOutputDType(outputDataType);

            auto expressionOutputs = Expression::outputs({{"feature_output", fout}});

            DynamicExpression::TensorMap stampInputs = inputs;
            stampInputs["feature_input"] = featureInputTensor;

            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
                stampInputs,
                {},
                outputs,
                {},
            };
        });
}

}  // namespace

bool FullyConnected::isFullyConnectedFloatingDataType(Tensor::DataType dataType) {
    return Thor::isFullyConnectedFloatingDataType(dataType);
}

std::string FullyConnected::dataTypeName(Tensor::DataType dataType) {
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

void FullyConnected::verifyFullyConnectedDataType(Tensor::DataType dataType, const std::string& what) {
    if (!isFullyConnectedFloatingDataType(dataType)) {
        throw std::invalid_argument("FullyConnected " + what + " must be one of fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32. Got " +
                                    dataTypeName(dataType) + ".");
    }
}

void FullyConnected::verifyFullyConnectedComputeDataType(Tensor::DataType dataType) {
    if (!cublasLtComputeTypeForFullyConnected(dataType).has_value()) {
        throw std::invalid_argument(
            "FullyConnected computeDataType must be fp32, fp16, or bf16 for Thor's current cuBLASLt floating GEMM path. Got " +
            dataTypeName(dataType) + ".");
    }
}

FullyConnected FullyConnected::Builder::build() {
    THOR_THROW_IF_FALSE(_network.has_value());
    THOR_THROW_IF_FALSE(!_featureInputs.empty());
    THOR_THROW_IF_FALSE(_numOutputFeatures.has_value());
    if (!_hasBias.has_value())
        _hasBias = false;
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
    if (!_computeDataType.has_value())
        _computeDataType = _featureInputs[0].getDataType();
    if (!_outputDataType.has_value())
        _outputDataType = _featureInputs[0].getDataType();

    verifyConfig();

    FullyConnected fullyConnected(_epilogue);

    fullyConnected.featureInputs = _featureInputs;
    fullyConnected.numOutputFeatures = _numOutputFeatures.value();

    fullyConnected.hasBias = _hasBias.value();
    if (_activation != nullptr)
        fullyConnected.activation = _activation;
    fullyConnected.weightsDataType = _weightsDataType.value();
    fullyConnected.computeDataType = _computeDataType.value();
    fullyConnected.outputDataType = _outputDataType.value();

    // Own parameter intent at the API layer. The stamped implementation layer is now the generic
    // CustomLayer, so there is no implementation FullyConnected class left to define parameters.
    std::shared_ptr<Initializer> weightsInitializer = _weightsInitializer->clone();
    std::shared_ptr<Initializer> biasInitializer = _hasBias.value() ? _biasInitializer->clone() : nullptr;
    const uint64_t inputFeatures = FullyConnected::checkedFeatureCount(_featureInputs.front().getDimensions(), "feature input");

    ParameterSpecification::Builder weightsParameterBuilder;
    weightsParameterBuilder.name("weights")
        .shape({inputFeatures, fullyConnected.numOutputFeatures})
        .dtype(fullyConnected.weightsDataType)
        .initializer(weightsInitializer)
        .trainable(true);
    if (_weightsOptimizer != nullptr)
        weightsParameterBuilder.optimizer(_weightsOptimizer);
    fullyConnected.addParameter(std::make_shared<ParameterSpecification>(weightsParameterBuilder.build()));

    if (fullyConnected.hasBias) {
        ParameterSpecification::Builder biasesParameterBuilder;
        biasesParameterBuilder.name("biases")
            .shape({fullyConnected.numOutputFeatures})
            .dtype(fullyConnected.weightsDataType)
            .initializer(biasInitializer)
            .trainable(true);
        if (_biasesOptimizer != nullptr)
            biasesParameterBuilder.optimizer(_biasesOptimizer);
        fullyConnected.addParameter(std::make_shared<ParameterSpecification>(biasesParameterBuilder.build()));
    }

    fullyConnected.initialized = true;

    for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
        Tensor out(fullyConnected.outputDataType, {fullyConnected.numOutputFeatures});
        fullyConnected.featureOutputs.push_back(out);

        fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = out;
        fullyConnected.inputTensorFromOutputTensor[out] = fullyConnected.featureInputs[i];
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
        FullyConnected::validateEpilogueExpression(_epilogue.value());
    }

    const Tensor::DataType inputDataType = _featureInputs.front().getDataType();
    const std::vector<uint64_t> inputDimensions = _featureInputs.front().getDimensions();
    FullyConnected::checkedFeatureCount(inputDimensions, "feature input");
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
        FullyConnected::checkedFeatureCount(featureInput.getDimensions(), "feature input " + std::to_string(i));
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
        buildFullyConnectedExpression(hasBias, placement, weightsDataType, computeDataType, outputDataType, activation, epilogue),
        placement,
        physicalParameters,
        inferenceOnly,
        getId(),
        false);
    physicalFullyConnected->setLayerName(getLayerType());

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
            serializableEpilogue = makeEpilogueDefinition(epilogue.value());
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

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition);
    }

    FullyConnected fullyConnected(epilogue);
    fullyConnected.numOutputFeatures = j.at("num_output_features").get<uint32_t>();
    fullyConnected.hasBias = j.at("has_bias").get<bool>();
    fullyConnected.weightsDataType = j.at("weights_data_type").get<Tensor::DataType>();
    fullyConnected.computeDataType = j.at("compute_data_type").get<Tensor::DataType>();
    fullyConnected.outputDataType = j.at("output_data_type").get<Tensor::DataType>();

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
