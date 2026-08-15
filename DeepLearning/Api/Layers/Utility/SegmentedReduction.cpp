#include "DeepLearning/Api/Layers/Utility/SegmentedReduction.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {

namespace {

std::vector<uint64_t> contiguousStrides(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> strides(dimensions.size(), 1);
    uint64_t stride = 1;
    for (size_t axis = dimensions.size(); axis-- > 0;) {
        strides[axis] = stride;
        if (dimensions[axis] != 0 && stride > std::numeric_limits<uint64_t>::max() / dimensions[axis]) {
            throw std::overflow_error("SegmentedReduction output stride overflows uint64_t.");
        }
        stride *= dimensions[axis];
    }
    return strides;
}

bool isFloatingPoint(DataType dataType) {
    switch (dataType) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
        case DataType::FP64:
            return true;
        default:
            return false;
    }
}

}  // namespace

const char* SegmentedReduction::typeName(Type type) {
    switch (type) {
        case Type::SUM: return "sum";
        case Type::MEAN: return "mean";
        case Type::MIN: return "min";
        case Type::MAX: return "max";
    }
    throw std::invalid_argument("Unknown SegmentedReduction type.");
}

SegmentedReduction::Type SegmentedReduction::typeFromName(const std::string& name) {
    if (name == "sum") return Type::SUM;
    if (name == "mean") return Type::MEAN;
    if (name == "min") return Type::MIN;
    if (name == "max") return Type::MAX;
    throw std::invalid_argument("Unknown SegmentedReduction type name: " + name + ".");
}

SegmentedReduction SegmentedReduction::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_reductionType.has_value()) {
        throw std::runtime_error("SegmentedReduction requires network, featureInput, and reductionType.");
    }
    if (_reductionType.value() == Type::MEAN && !isFloatingPoint(_featureInput->getValuesDataType())) {
        throw std::invalid_argument("SegmentedReduction mean requires floating-point ragged values.");
    }

    if (_featureInput->getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("SegmentedReduction batch size exceeds Thor's uint32 placement capacity.");
    }
    std::vector<uint64_t> outputDimensions = _featureInput->getTrailingDimensions();
    if (outputDimensions.empty()) outputDimensions = {1};

    SegmentedReduction reduction;
    reduction.raggedFeatureInput = _featureInput.value();
    reduction.reductionType = _reductionType.value();
    reduction.featureInputs = {_featureInput->getValues(), _featureInput->getOffsets()};
    reduction.featureOutputs = {Tensor(_featureInput->getValuesDataType(), outputDimensions)};
    reduction.initialized = true;
    reduction.addToNetwork(_network.value());
    return reduction;
}

std::vector<Tensor> SegmentedReduction::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) {
        return {};
    }
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void SegmentedReduction::informThatInputConnectionMade(Tensor inputTensor) {
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void SegmentedReduction::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int SegmentedReduction::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
    }
    if (!featureOutputs.empty() && connectingTensor == featureOutputs.front()) return 0;
    throw std::runtime_error("Tensor is not connected to this SegmentedReduction layer.");
}

uint64_t SegmentedReduction::getOutputTensorBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    const uint64_t perExampleBytes = featureOutputs.front().getTotalSizeInBytes();
    if (batchSize != 0 && perExampleBytes > std::numeric_limits<uint64_t>::max() / batchSize) {
        throw std::overflow_error("SegmentedReduction output size overflows uint64_t.");
    }
    return perExampleBytes * batchSize;
}

std::shared_ptr<ThorImplementation::Layer> SegmentedReduction::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInputs.size() == 2);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    (void)getConnectionType(connectingApiTensor);

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;
    using ThorImplementation::RaggedExpression;

    RaggedExpression input = RaggedExpression::input("feature_input", "feature_offsets", raggedFeatureInput.getDescriptor());
    Expression output = [&]() -> Expression {
        switch (reductionType) {
            case Type::SUM: return input.segment_sum();
            case Type::MEAN: return input.segment_mean();
            case Type::MIN: return input.segment_min();
            case Type::MAX: return input.segment_max();
        }
        throw std::invalid_argument("Unknown SegmentedReduction type.");
    }();

    const std::vector<uint64_t> outputDimensions = featureOutputs.front().getDimensions();
    std::vector<uint64_t> physicalOutputDimensions;
    physicalOutputDimensions.reserve(outputDimensions.size() + 1);
    physicalOutputDimensions.push_back(raggedFeatureInput.getBatchSize());
    physicalOutputDimensions.insert(physicalOutputDimensions.end(), outputDimensions.begin(), outputDimensions.end());
    output = output.stridedView(physicalOutputDimensions, contiguousStrides(physicalOutputDimensions), 0);

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", output}}));
    auto physicalLayer = std::make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        std::vector<std::string>{"feature_input", "feature_offsets"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        getId(),
        std::vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor>{
            {featureOutputs.front().getDataType(), outputDimensions, false}},
        false,
        false,
        std::vector<bool>{true, true},
        static_cast<uint32_t>(raggedFeatureInput.getBatchSize()));
    physicalLayer->setLayerName(getLayerType());
    return physicalLayer;
}

uint64_t SegmentedReduction::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize,
    ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)batchSize;
    (void)tensorPlacement;
    THOR_THROW_IF_FALSE(featureInputs.size() == 2);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    const uint64_t perExampleOutputBytes = featureOutputs.front().getTotalSizeInBytes();
    if (raggedFeatureInput.getBatchSize() != 0 &&
        perExampleOutputBytes > std::numeric_limits<uint64_t>::max() / raggedFeatureInput.getBatchSize()) {
        throw std::overflow_error("SegmentedReduction output memory requirement overflows uint64_t.");
    }
    uint64_t bytes = perExampleOutputBytes * raggedFeatureInput.getBatchSize();
    for (const Tensor& input : featureInputs) {
        if (bytes > std::numeric_limits<uint64_t>::max() - input.getTotalSizeInBytes()) {
            throw std::overflow_error("SegmentedReduction memory requirement overflows uint64_t.");
        }
        bytes += input.getTotalSizeInBytes();
    }
    return bytes;
}

json SegmentedReduction::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInputs.size() == 2);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "segmented_reduction"},
                {"reduction_type", typeName(reductionType)},
                {"ragged_feature_input", raggedFeatureInput.architectureJson()},
                {"feature_output", featureOutputs.front().architectureJson()}};
}

void SegmentedReduction::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in SegmentedReduction::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "segmented_reduction") {
        throw std::runtime_error("Layer type mismatch in SegmentedReduction::deserialize: " + j.at("layer_type").get<std::string>());
    }

    const json& inputJson = j.at("ragged_feature_input");
    Tensor values = network->getApiTensorByOriginalId(inputJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(inputJson.at("offsets").at("id").get<uint64_t>());
    RaggedTensor raggedInput(values, offsets);
    if (raggedInput.getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("SegmentedReduction serialized batch size exceeds Thor's uint32 placement capacity.");
    }
    if (raggedInput.getBatchSize() != inputJson.at("batch_size").get<uint64_t>() ||
        raggedInput.getMaxTotalValues() != inputJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error("SegmentedReduction serialized ragged input metadata does not match reconstructed tensors.");
    }

    Tensor output = Tensor::deserialize(j.at("feature_output"));
    std::vector<uint64_t> expectedDimensions = raggedInput.getTrailingDimensions();
    if (expectedDimensions.empty()) expectedDimensions = {1};
    if (output.getDimensions() != expectedDimensions || output.getDataType() != values.getDataType()) {
        throw std::runtime_error("SegmentedReduction serialized output descriptor does not match the ragged input.");
    }

    SegmentedReduction reduction;
    reduction.raggedFeatureInput = raggedInput;
    reduction.reductionType = typeFromName(j.at("reduction_type").get<std::string>());
    if (reduction.reductionType == Type::MEAN && !isFloatingPoint(values.getDataType())) {
        throw std::runtime_error("SegmentedReduction serialized mean requires floating-point ragged values.");
    }
    reduction.featureInputs = {values, offsets};
    reduction.featureOutputs = {output};
    reduction.initialized = true;
    reduction.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("segmented_reduction", &Thor::SegmentedReduction::deserialize);
    return true;
}();
}  // namespace
