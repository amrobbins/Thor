#include "DeepLearning/Api/Layers/Utility/Add.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
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
    if (!left.hasMaxValuesPerRow() || !right.hasMaxValuesPerRow() ||
        left.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
        right.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
        left.getValues() == right.getValues() ||
        !left.sharesPartitionWith(right) ||
        left.getMaxValuesPerRow() != right.getMaxValuesPerRow() ||
        left.getOffsetsDataType() != right.getOffsetsDataType() ||
        left.getValuesDataType() != right.getValuesDataType() ||
        left.getValuesDimensions() != right.getValuesDimensions()) {
        throw std::invalid_argument("Add requires distinct values with identical logical geometry and a shared row partition.");
    }
}

uint64_t checkedAdd(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error("Add byte sum overflow.");
    return a + b;
}
uint64_t placedBytes(const RaggedTensor& r, uint32_t batch) {
    const uint64_t n = r.getValues().getTotalSizeInBytes();
    THOR_THROW_IF_FALSE(r.hasMaxValuesPerRow());
    if (batch == 0 || n > std::numeric_limits<uint64_t>::max() / batch)
        throw std::overflow_error("Add output placement byte product overflow.");
    return n * batch;
}
// Deserialization gives a tensor a new live ID while retaining the saved original
// ID. Serialized references must be compared to that canonical original ID,
// rather than to a new architectureJson() containing the live ID.
void checkCanonicalTensorReference(const json& serialized, const Tensor& tensor) {
    if (serialized.at("version").get<std::string>() != "1.0.0" ||
        serialized.at("id").get<uint64_t>() != tensor.getOriginalId() ||
        serialized.at("dimensions").get<std::vector<uint64_t>>() != tensor.getDimensions() ||
        serialized.at("data_type").get<DataType>() != tensor.getDataType()) {
        throw std::runtime_error("Add canonical reference has forged semantic tensor geometry.");
    }
}

void checkCanonicalReference(const json& serialized, const RaggedTensor& r) {
    if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(serialized, "Add") !=
        SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) return;
    if (serialized.contains("batch_size") || serialized.contains("max_total_values")) {
        throw std::runtime_error("Add canonical references must not contain archived physical batch metadata.");
    }
    checkCanonicalTensorReference(serialized.at("values"), r.getValues());
    checkCanonicalTensorReference(serialized.at("offsets"), r.getRowPartitionToken());
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
        add.raggedOutput = _raggedLeft->withValues(outputValues);
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
    if (raggedOutput.has_value()) return placedBytes(raggedOutput.value(), batchSize);
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
        // The stamp can be driven by either values branch or the managed
        // structural partition carrier; runtime metadata is authoritative.
        const RaggedTensor& drivenInput = connectingApiTensor == raggedRight->getValues()
            ? raggedRight.value() : raggedLeft.value();
        const auto driven = SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            drivenInput, drivingLayer, "Add", connectingApiTensor == raggedLeft->getOffsets());
        const auto leftDescriptor = raggedLeft->getDescriptorForPlacement(driven.getBatchSize());
        const auto rightDescriptor = raggedRight->getDescriptorForPlacement(driven.getBatchSize());
        if (driven != (connectingApiTensor == raggedRight->getValues() ? rightDescriptor : leftDescriptor) ||
            leftDescriptor.getValuesDescriptor() != rightDescriptor.getValuesDescriptor() ||
            leftDescriptor.getRowPartition() != rightDescriptor.getRowPartition()) {
            throw std::runtime_error("Add physical descriptor disagrees with both neutral operands.");
        }
        Expression offsets = Expression::input("offsets");
        RaggedExpression left(Expression::input("left"), offsets, leftDescriptor,
                              ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        RaggedExpression right(Expression::input("right"), offsets, rightDescriptor,
                               ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
        RaggedExpression output = left + right;
        ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
            Expression::outputs({{"feature_output", output.getValues()}}));
        auto physical = std::make_shared<ThorImplementation::RaggedCustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            std::vector<std::string>{"left", "right", "offsets"},
            std::vector<std::string>{"feature_output"},
            placement,
            inferenceOnly,
            leftDescriptor.getMaxTotalValues(),
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
            const uint64_t inputBytes = input == raggedLeft->getOffsets()
                ? input.getTotalSizeInBytes()  // managed DEVICE_ACTIVE_COUNT [1], never [B+1]
                : placedBytes(input == raggedLeft->getValues() ? raggedLeft.value() : raggedRight.value(), batchSize);
            bytes = checkedAdd(bytes, inputBytes);
        }
    }
    return bytes;
}

json Add::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    json j{{"factory", Layer::Factory::Layer.value()},
           {"version", raggedLeft.has_value() ? "1.1.0" : getLayerVersion()},
           {"layer_type", "add"},
           {"feature_output", featureOutputs.front().architectureJson()}};
    if (raggedLeft.has_value()) {
        j["use_ragged"] = true;
        j["ragged_left"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedLeft.value());
        j["ragged_right"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedRight.value());
        j["ragged_output"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedOutput.value());
    } else {
        j["left"] = featureInputs[0].architectureJson();
        j["right"] = featureInputs[1].architectureJson();
    }
    return j;
}

void Add::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw std::runtime_error("Unsupported version in Add::deserialize: " + version);
    if (j.at("layer_type").get<std::string>() != "add") {
        throw std::runtime_error("Layer type mismatch in Add::deserialize: " + j.at("layer_type").get<std::string>());
    }

    Add add;
    if (j.value("use_ragged", false)) {
        const json& leftJson = j.at("ragged_left");
        const json& rightJson = j.at("ragged_right");
        add.raggedLeft = SegmentedPrimitiveDetail::reconstructInput(leftJson, network, "Add");
        add.raggedRight = SegmentedPrimitiveDetail::reconstructInput(rightJson, network, "Add");
        // Historical 1.0.0 references use packed dimensions; the reference
        // grammar does not select a model geometry. Resolve both from the graph.
        requireCompatibleRagged(add.raggedLeft.value(), add.raggedRight.value());
        checkCanonicalReference(leftJson, add.raggedLeft.value());
        checkCanonicalReference(rightJson, add.raggedRight.value());
        const json& outputJson = j.at("ragged_output");
        Tensor outputValues = SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
            j.at("feature_output"), outputJson, add.raggedLeft.value(), "Add");
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(outputJson, leftJson, add.raggedLeft.value(), "Add");
        if (outputJson.at("offsets").at("id").get<uint64_t>() != leftJson.at("offsets").at("id").get<uint64_t>() ||
            outputJson.at("values").at("id").get<uint64_t>() != outputValues.getOriginalId())
            throw std::runtime_error("Add output must preserve canonical partition and output values identities.");
        if (outputJson.value("ragged_rank", 1U) != 1 ||
            outputValues.getDimensions() != add.raggedLeft->getValuesDimensions() ||
            outputValues.getDataType() != add.raggedLeft->getValuesDataType() ||
            outputValues.getOriginalId() == add.raggedLeft->getValues().getOriginalId() ||
            outputValues.getOriginalId() == add.raggedRight->getValues().getOriginalId()) {
            throw std::runtime_error("Add output must have distinct logical values and the shared partition.");
        }
        checkCanonicalReference(outputJson, add.raggedLeft->withValues(outputValues));
        add.raggedOutput = add.raggedLeft->withValues(outputValues);
        add.featureInputs = {add.raggedLeft->getValues(), add.raggedRight->getValues(), add.raggedLeft->getOffsets()};
        add.featureOutputs = {outputValues};
    } else {
        if (version != "1.0.0") throw std::runtime_error("Add neutral version requires ragged operands.");
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
