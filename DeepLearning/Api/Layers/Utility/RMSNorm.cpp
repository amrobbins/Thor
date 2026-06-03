#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"

#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {

uint64_t checkedProductForRmsNorm(const std::vector<uint64_t>& dims, const std::string& what) {
    uint64_t product = 1;
    for (uint64_t dim : dims) {
        if (dim == 0) {
            throw std::runtime_error("RMSNorm " + what + " dimensions must be non-zero.");
        }
        if (product > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error("RMSNorm " + what + " product overflows uint64_t.");
        }
        product *= dim;
    }
    return product;
}

bool isSwishEpilogueExpression(const ThorImplementation::Expression& epilogue) {
    Swish swish;
    ThorImplementation::Expression reference = swish.toExpression(RMSNorm::epilogueInput());
    return LayerEpilogue::hasSameCanonicalForm(epilogue, reference, RMSNorm::epilogueInputName(), RMSNorm::epilogueOutputName(), "RMSNorm");
}

ThorImplementation::DynamicExpression buildRmsNormExpression(ThorImplementation::TensorPlacement placement,
                                                             std::vector<uint64_t> normalizedShape,
                                                             uint64_t hidden,
                                                             double epsilon,
                                                             DataType parameterDataType,
                                                             std::optional<ThorImplementation::Expression> epilogue,
                                                             bool inferenceOnly) {
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;
    using ThorImplementation::TensorDescriptor;

    const bool epilogueIsSwish = epilogue.has_value() && isSwishEpilogueExpression(epilogue.value());

    return DynamicExpression(
        {"feature_input", "weights"},
        {"feature_output"},
        [placement,
         normalizedShape = std::move(normalizedShape),
         hidden,
         epsilon,
         parameterDataType,
         epilogue = std::move(epilogue),
         epilogueIsSwish,
         inferenceOnly](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            Tensor featureInputTensor = inputs.at("feature_input");
            const Tensor& weightsTensor = inputs.at("weights");
            const std::vector<uint64_t> originalInputDims = featureInputTensor.getDimensions();
            const DataType inputDataType = featureInputTensor.getDataType();

            if (featureInputTensor.getPlacement() != placement) {
                throw std::runtime_error("RMSNorm feature input tensor placement does not match the layer placement.");
            }
            if (weightsTensor.getPlacement() != placement) {
                throw std::runtime_error("RMSNorm weights tensor placement does not match the layer placement.");
            }
            if (weightsTensor.getDataType() != parameterDataType) {
                throw std::runtime_error("RMSNorm weights tensor dtype does not match parameterDataType.");
            }
            if (weightsTensor.getDimensions().size() != 1 || weightsTensor.getDimensions()[0] != hidden) {
                throw std::runtime_error("RMSNorm weights tensor must have shape [normalized_feature_count].");
            }
            if (originalInputDims.size() < normalizedShape.size()) {
                throw std::runtime_error("RMSNorm normalizedShape rank cannot exceed feature input rank.");
            }
            const size_t normalizedOffset = originalInputDims.size() - normalizedShape.size();
            for (size_t i = 0; i < normalizedShape.size(); ++i) {
                if (originalInputDims[normalizedOffset + i] != normalizedShape[i]) {
                    throw std::runtime_error("RMSNorm normalizedShape must match trailing feature input dimensions.");
                }
            }

            const uint64_t outer = checkedProductForRmsNorm(
                std::vector<uint64_t>(originalInputDims.begin(), originalInputDims.begin() + static_cast<std::ptrdiff_t>(normalizedOffset)),
                "outer");
            featureInputTensor.reshape({outer, hidden});

            if (outputs.contains("feature_output")) {
                const Tensor& featureOutputTensor = outputs.at("feature_output");
                if (featureOutputTensor.getPlacement() != placement) {
                    throw std::runtime_error("RMSNorm feature output tensor placement does not match the layer placement.");
                }
                if (featureOutputTensor.getDataType() != inputDataType) {
                    throw std::runtime_error("RMSNorm feature output tensor dtype must match the feature input dtype.");
                }
                if (featureOutputTensor.getDimensions() != originalInputDims) {
                    throw std::runtime_error("RMSNorm feature output tensor dimensions must match the feature input dimensions.");
                }
            }

            Expression fin = Expression::input("feature_input", inputDataType, inputDataType);
            Expression weights = Expression::input("weights", parameterDataType, parameterDataType);
            Expression fout = Expression::rmsNorm(fin, weights, hidden, epsilon, DataType::FP32, inputDataType);
            const bool useCudnnSwishFusion = epilogueIsSwish && parameterDataType == DataType::BF16 &&
                                             inputDataType == DataType::BF16;
            if (useCudnnSwishFusion) {
                if (!inferenceOnly) {
                    throw std::runtime_error(
                        "RMSNorm Swish epilogue can use cuDNN Frontend RMSNorm + SiLU only for inference; "
                        "training should use fp32 RMSNorm weights so the Swish epilogue can run as a separate expression.");
                }
                ThorImplementation::PhysicalExpression physical = fout.expression();
                if (physical.output_node >= physical.nodes.size() || physical.nodes[physical.output_node].op != ThorImplementation::ExprOp::RMSNORM) {
                    throw std::runtime_error("RMSNorm internal fusion rewrite expected an RMSNORM output node.");
                }
                physical.nodes[physical.output_node].rms_norm_fused_activation =
                    ThorImplementation::CudnnRmsNormFusedActivation::SWISH;
                auto fusedPhysical = std::make_shared<ThorImplementation::PhysicalExpression>(std::move(physical));
                fout = Expression::fromPhysicalNode(fusedPhysical, fusedPhysical->output_node);
            } else if (epilogue.has_value()) {
                fout = RMSNorm::applyEpilogue(fout, epilogue.value());
            }
            if (featureInputTensor.getDimensions() != originalInputDims) {
                fout = fout.reshape(originalInputDims);
            }
            fout = fout.withOutputDType(inputDataType);

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

bool RMSNorm::isRMSNormInputDataType(DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

uint64_t RMSNorm::checkedFeatureCount(const vector<uint64_t>& shape, const string& what) {
    if (shape.empty()) {
        throw invalid_argument("RMSNorm " + what + " must contain at least one dimension.");
    }
    uint64_t count = 1;
    for (uint64_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("RMSNorm " + what + " dimensions must be non-zero.");
        }
        if (count > numeric_limits<uint64_t>::max() / dim) {
            throw invalid_argument("RMSNorm " + what + " feature count overflows uint64_t.");
        }
        count *= dim;
    }
    return count;
}

void RMSNorm::validateNormalizedShapeForInput(const vector<uint64_t>& inputDims, const vector<uint64_t>& normalizedShape) {
    if (inputDims.empty()) {
        throw invalid_argument("RMSNorm feature input must have at least one feature dimension.");
    }
    if (inputDims.size() < normalizedShape.size()) {
        throw invalid_argument("RMSNorm normalizedShape rank cannot exceed feature input rank.");
    }
    const size_t offset = inputDims.size() - normalizedShape.size();
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
        if (inputDims[offset + i] != normalizedShape[i]) {
            throw invalid_argument("RMSNorm normalizedShape must match trailing feature input dimensions.");
        }
    }
}

RMSNorm RMSNorm::Builder::build() {
    if (_featureInputs.empty()) {
        throw invalid_argument("RMSNorm::Builder requires at least one featureInput().");
    }
    if (_normalizedShape.empty()) {
        const vector<uint64_t> dims = _featureInputs.front().getDimensions();
        if (dims.empty()) {
            throw invalid_argument("RMSNorm feature input must have at least one feature dimension.");
        }
        _normalizedShape = {dims.back()};
    }
    if (!_epsilon.has_value())
        _epsilon = 1.0e-5;
    if (!_parameterDataType.has_value()) {
        _parameterDataType = DataType::FP32;
    }
    if (_weightsInitializer == nullptr)
        _weightsInitializer = UniformRandom::Builder().minValue(1.0f).maxValue(1.0f).build();

    verifyConfig();

    RMSNorm layer(_epilogue);
    layer.featureInputs = _featureInputs;
    layer.normalizedShape = _normalizedShape;
    layer.epsilon = _epsilon.value();
    layer.parameterDataType = _parameterDataType.value();
    const uint64_t hidden = RMSNorm::checkedFeatureCount(layer.normalizedShape, "normalizedShape");

    ParameterSpecification::Builder weightsBuilder;
    weightsBuilder.name("weights").shape({hidden}).dtype(layer.parameterDataType).initializer(_weightsInitializer).trainable(true);
    if (_weightsOptimizer != nullptr)
        weightsBuilder.optimizer(_weightsOptimizer);
    layer.addParameter(make_shared<ParameterSpecification>(weightsBuilder.build()));

    layer.initialized = true;

    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        Tensor out = layer.featureInputs[i].clone();
        layer.featureOutputs.push_back(out);
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = out;
        layer.inputTensorFromOutputTensor[out] = layer.featureInputs[i];
    }

    layer.addToNetwork(_network.value());
    return layer;
}

void RMSNorm::Builder::verifyConfig() const {
    if (!_network.has_value()) {
        throw invalid_argument("RMSNorm::Builder requires network().");
    }
    if (_featureInputs.empty()) {
        throw invalid_argument("RMSNorm::Builder requires featureInput().");
    }
    checkedFeatureCount(_normalizedShape, "normalizedShape");
    if (!_epsilon.has_value() || !(_epsilon.value() > 0.0)) {
        throw invalid_argument("RMSNorm epsilon must be > 0.");
    }
    const DataType inputDataType = _featureInputs.front().getDataType();
    const bool swishEpilogue = _epilogue.has_value() && isSwishEpilogueExpression(_epilogue.value());
    if (_epilogue.has_value()) {
        RMSNorm::validateEpilogueExpression(_epilogue.value());
    }
    if (_parameterDataType.value() == DataType::BF16) {
        if (!swishEpilogue) {
            throw invalid_argument(
                "RMSNorm bf16 weights are only supported for the cuDNN Frontend RMSNorm + Swish epilogue inference fusion; "
                "use fp32 weights for standard RMSNorm or non-Swish epilogues.");
        }
        if (inputDataType != DataType::BF16) {
            throw invalid_argument(
                "RMSNorm Swish epilogue fusion with bf16 weights requires bf16 feature inputs; use fp32 weights for generic epilogue execution.");
        }
    } else if (_parameterDataType.value() != DataType::FP32) {
        throw invalid_argument("RMSNorm currently requires fp32 weights, except bf16 weights for the Swish epilogue inference fusion.");
    }
    if (!RMSNorm::isRMSNormInputDataType(inputDataType)) {
        throw invalid_argument("RMSNorm feature input dtype must be fp16, bf16, or fp32.");
    }
    const vector<uint64_t> inputDims = _featureInputs.front().getDimensions();
    RMSNorm::validateNormalizedShapeForInput(inputDims, _normalizedShape);
    for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
        if (!_featureInputs[i].isInitialized()) {
            throw invalid_argument("RMSNorm feature input is not initialized.");
        }
        if (_featureInputs[i].getDataType() != inputDataType) {
            throw invalid_argument("RMSNorm all feature inputs must have the same dtype.");
        }
        if (_featureInputs[i].getDimensions() != inputDims) {
            throw invalid_argument("RMSNorm all feature inputs must have the same dimensions.");
        }
    }
}

shared_ptr<ThorImplementation::Layer> RMSNorm::stamp(ThorImplementation::TensorPlacement placement,
                                                     shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

    vector<shared_ptr<ThorImplementation::PhysicalParameter>> physicalParameters;
    for (const auto& parameter : getParameters()) {
        THOR_THROW_IF_FALSE(parameter != nullptr);
        physicalParameters.push_back(parameter->stamp());
    }

    const uint64_t hidden = RMSNorm::checkedFeatureCount(normalizedShape, "normalizedShape");
    shared_ptr<ThorImplementation::CustomLayer> physicalRmsNorm = make_shared<ThorImplementation::CustomLayer>(
        buildRmsNormExpression(placement, normalizedShape, hidden, epsilon, parameterDataType, epilogue, inferenceOnly),
        placement,
        physicalParameters,
        inferenceOnly,
        getId());
    physicalRmsNorm->setLayerName(getLayerType());
    return physicalRmsNorm;
}

json RMSNorm::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "rms_norm";
    j["layer_name"] = string("layer") + to_string(getId());
    j["normalized_shape"] = normalizedShape;
    j["epsilon"] = epsilon;
    j["parameter_data_type"] = parameterDataType;
    if (epilogue.has_value()) {
        if (!serializableEpilogue.has_value())
            serializableEpilogue = makeEpilogueDefinition(epilogue.value());
        j["epilogue"] = serializableEpilogue.value().architectureJson();
    } else {
        j["epilogue"] = nullptr;
    }

    json inputs = json::array();
    for (uint32_t i = 0; i < featureInputs.size(); ++i)
        inputs.push_back(featureInputs[i].architectureJson());
    j["inputs"] = inputs;

    json outputs = json::array();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i)
        outputs.push_back(featureOutputs[i].architectureJson());
    j["outputs"] = outputs;

    j["parameters"] = getParametersArchitectureJson()["parameters"];
    return j;
}

json RMSNorm::serialize(thor_file::TarWriter& archiveWriter,
                        Stream stream,
                        bool saveOptimizerState,
                        ThorImplementation::StampedNetwork& stampedNetwork) const {
    json j = architectureJson();
    Parameterizable::serializeParameters(j["parameters"], archiveWriter, stream, saveOptimizerState, stampedNetwork, "layer" + to_string(getId()));
    return j;
}

void RMSNorm::deserialize(shared_ptr<thor_file::TarReader>& archiveReader, const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in RMSNorm::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "rms_norm")
        throw runtime_error("Layer type mismatch in RMSNorm::deserialize: " + j.at("layer_type").get<string>());

    std::optional<ThorImplementation::Expression> epilogue = std::nullopt;
    if (j.contains("epilogue") && !j.at("epilogue").is_null()) {
        ThorImplementation::ExpressionDefinition epilogueDefinition =
            ThorImplementation::ExpressionDefinition::deserialize(j.at("epilogue"));
        epilogue = epilogueExpressionFromDefinition(epilogueDefinition);
    } else if (j.contains("fused_activation") &&
               ThorImplementation::cudnnRmsNormFusedActivationFromString(j.at("fused_activation").get<string>()) ==
                   ThorImplementation::CudnnRmsNormFusedActivation::SWISH) {
        Swish swish;
        epilogue = swish.toExpression(RMSNorm::epilogueInput());
    }

    RMSNorm layer(epilogue);
    layer.normalizedShape = j.at("normalized_shape").get<vector<uint64_t>>();
    layer.epsilon = j.at("epsilon").get<double>();
    layer.parameterDataType = j.at("parameter_data_type").get<DataType>();

    for (const json& inputJson : j.at("inputs")) {
        const uint64_t originalTensorId = inputJson.at("id").get<uint64_t>();
        layer.featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }
    for (const json& outputJson : j.at("outputs")) {
        layer.featureOutputs.push_back(Tensor::deserialize(outputJson, archiveReader.get()));
    }
    if (layer.featureInputs.size() != layer.featureOutputs.size()) {
        throw runtime_error("RMSNorm deserialize expected equal numbers of inputs and outputs.");
    }
    for (uint32_t i = 0; i < layer.featureInputs.size(); ++i) {
        layer.outputTensorFromInputTensor[layer.featureInputs[i]] = layer.featureOutputs[i];
        layer.inputTensorFromOutputTensor[layer.featureOutputs[i]] = layer.featureInputs[i];
    }

    if (j.contains("parameters")) {
        const json& parametersJson = j.at("parameters");
        if (!parametersJson.is_object()) {
            throw runtime_error("RMSNorm parameters must be an object keyed by parameter name.");
        }
        for (auto it = parametersJson.begin(); it != parametersJson.end(); ++it) {
            ParameterSpecification parameter = ParameterSpecification::deserialize(it.value(), archiveReader);
            layer.addParameter(make_shared<ParameterSpecification>(std::move(parameter)));
        }
    }
    if (!layer.hasParameter("weights")) {
        throw runtime_error("RMSNorm deserialize did not find required weights parameter.");
    }

    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::TrainableLayer::register_layer("rms_norm", &Thor::RMSNorm::deserialize);
    return true;
}();
}  // namespace
