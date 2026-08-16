#include "DeepLearning/Api/Layers/Utility/Add.h"

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

namespace {

uint64_t elementsPerValue(const std::vector<uint64_t>& dimensions) {
    uint64_t elements = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension == 0 || elements > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::overflow_error("Add ragged elements-per-value overflow.");
        }
        elements *= dimension;
    }
    return elements;
}

void requireCompatibleDense(const Tensor& left, const Tensor& right) {
    if (left.getDimensions() != right.getDimensions()) {
        throw std::invalid_argument("Add inputs must have identical dimensions.");
    }
    if (left.getDataType() != right.getDataType()) {
        throw std::invalid_argument("Add inputs must have identical data types.");
    }
}

void requireCompatibleRagged(const RaggedTensor& left, const RaggedTensor& right) {
    if (left.getOffsets() != right.getOffsets()) {
        throw std::invalid_argument("Add RaggedTensor inputs must share the exact same offsets tensor.");
    }
    if (left.getDescriptor() != right.getDescriptor()) {
        throw std::invalid_argument("Add RaggedTensor inputs must have identical row partitions, value shapes, and dtypes.");
    }
}

}  // namespace

Add Add::Builder::build() {
    if (!_network.has_value() || !_left.has_value() || !_right.has_value()) {
        throw std::runtime_error("Add requires network, left, and right inputs.");
    }
    const bool leftRagged = _raggedLeft.has_value();
    const bool rightRagged = _raggedRight.has_value();
    if (leftRagged != rightRagged) {
        throw std::invalid_argument("Add cannot mix dense Tensor and RaggedTensor inputs.");
    }

    Add add;
    if (leftRagged) {
        requireCompatibleRagged(_raggedLeft.value(), _raggedRight.value());
        add.raggedLeft = _raggedLeft.value();
        add.raggedRight = _raggedRight.value();
        add.featureInputs = {_raggedLeft->getValues(), _raggedRight->getValues(), _raggedLeft->getOffsets()};
        Tensor outputValues(_raggedLeft->getValuesDataType(), _raggedLeft->getValuesDimensions());
        add.featureOutputs = {outputValues};
        add.raggedOutput = RaggedTensor(outputValues, _raggedLeft->getOffsets());
    } else {
        requireCompatibleDense(_left.value(), _right.value());
        add.featureInputs = {_left.value(), _right.value()};
        add.featureOutputs = {Tensor(_left->getDataType(), _left->getDimensions())};
    }

    for (const Tensor& input : add.featureInputs) {
        add.outputTensorFromInputTensor[input] = add.featureOutputs.front();
    }
    add.initialized = true;
    add.addToNetwork(_network.value());
    return add;
}

std::vector<Tensor> Add::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void Add::informThatInputConnectionMade(Tensor inputTensor) {
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void Add::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int Add::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
    }
    if (!featureOutputs.empty() && connectingTensor == featureOutputs.front()) return 0;
    throw std::runtime_error("Tensor is not connected to this Add layer.");
}

uint64_t Add::getOutputTensorBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    if (raggedOutput.has_value()) return featureOutputs.front().getTotalSizeInBytes();
    const uint64_t perExample = featureOutputs.front().getTotalSizeInBytes();
    if (batchSize != 0 && perExample > std::numeric_limits<uint64_t>::max() / batchSize) {
        throw std::overflow_error("Add output size overflows uint64_t.");
    }
    return perExample * batchSize;
}

std::shared_ptr<ThorImplementation::Layer> Add::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    (void)getConnectionType(connectingApiTensor);

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;
    using ThorImplementation::RaggedExpression;

    if (raggedLeft.has_value()) {
        THOR_THROW_IF_FALSE(raggedRight.has_value() && raggedOutput.has_value());
        Expression offsets = Expression::input("offsets");
        RaggedExpression left(Expression::input("left"), offsets, raggedLeft->getDescriptor());
        RaggedExpression right(Expression::input("right"), offsets, raggedRight->getDescriptor());
        RaggedExpression output = left + right;
        ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
            Expression::outputs({{"feature_output", output.getValues()}}));
        auto physical = std::make_shared<ThorImplementation::RaggedCustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            std::vector<std::string>{"left", "right", "offsets"},
            std::vector<std::string>{"feature_output"},
            placement,
            inferenceOnly,
            raggedLeft->getMaxTotalValues(),
            std::vector<uint64_t>{elementsPerValue(raggedLeft->getTrailingDimensions()),
                                  elementsPerValue(raggedRight->getTrailingDimensions())},
            elementsPerValue(raggedOutput->getTrailingDimensions()),
            std::vector<uint32_t>{0, 1},
            2,
            getId());
        physical->setLayerName(getLayerType());
        return physical;
    }

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs(
        {{"feature_output", Expression::input("left") + Expression::input("right")}}));
    auto physical = std::make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        std::vector<std::string>{"left", "right"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        getId());
    physical->setLayerName(getLayerType());
    return physical;
}

uint64_t Add::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize,
    ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    uint64_t bytes = getOutputTensorBytes(batchSize);
    if (raggedOutput.has_value()) {
        for (const Tensor& input : featureInputs) {
            if (bytes > std::numeric_limits<uint64_t>::max() - input.getTotalSizeInBytes()) {
                throw std::overflow_error("Ragged Add memory requirement overflows uint64_t.");
            }
            bytes += input.getTotalSizeInBytes();
        }
    }
    return bytes;
}

json Add::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    json j{{"factory", Layer::Factory::Layer.value()},
           {"version", getLayerVersion()},
           {"layer_type", "add"},
           {"feature_output", featureOutputs.front().architectureJson()}};
    if (raggedLeft.has_value()) {
        j["use_ragged"] = true;
        j["ragged_left"] = raggedLeft->architectureJson();
        j["ragged_right"] = raggedRight->architectureJson();
        j["ragged_output"] = raggedOutput->architectureJson();
    } else {
        j["left"] = featureInputs[0].architectureJson();
        j["right"] = featureInputs[1].architectureJson();
    }
    return j;
}

void Add::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in Add::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "add") {
        throw std::runtime_error("Layer type mismatch in Add::deserialize: " + j.at("layer_type").get<std::string>());
    }

    Add add;
    if (j.value("use_ragged", false)) {
        auto restoreRagged = [&](const json& r) {
            Tensor values = network->getApiTensorByOriginalId(r.at("values").at("id").get<uint64_t>());
            Tensor offsets = network->getApiTensorByOriginalId(r.at("offsets").at("id").get<uint64_t>());
            return RaggedTensor(values, offsets);
        };
        add.raggedLeft = restoreRagged(j.at("ragged_left"));
        add.raggedRight = restoreRagged(j.at("ragged_right"));
        requireCompatibleRagged(add.raggedLeft.value(), add.raggedRight.value());
        Tensor outputValues = Tensor::deserialize(j.at("feature_output"));
        add.raggedOutput = RaggedTensor(outputValues, add.raggedLeft->getOffsets());
        const json& outputJson = j.at("ragged_output");
        if (outputJson.at("offsets").at("id").get<uint64_t>() !=
                j.at("ragged_left").at("offsets").at("id").get<uint64_t>() ||
            outputJson.at("values").at("id").get<uint64_t>() !=
                j.at("feature_output").at("id").get<uint64_t>()) {
            throw std::runtime_error("Add serialized ragged output does not preserve the left/right row partition.");
        }
        add.featureInputs = {add.raggedLeft->getValues(), add.raggedRight->getValues(), add.raggedLeft->getOffsets()};
        add.featureOutputs = {outputValues};
    } else {
        Tensor left = network->getApiTensorByOriginalId(j.at("left").at("id").get<uint64_t>());
        Tensor right = network->getApiTensorByOriginalId(j.at("right").at("id").get<uint64_t>());
        requireCompatibleDense(left, right);
        add.featureInputs = {left, right};
        add.featureOutputs = {Tensor::deserialize(j.at("feature_output"))};
    }
    for (const Tensor& input : add.featureInputs) add.outputTensorFromInputTensor[input] = add.featureOutputs.front();
    add.initialized = true;
    add.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("add", &Thor::Add::deserialize);
    return true;
}();
}
