#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <limits>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace Thor {
namespace {
void validateCanonicalTensorReference(const json& serialized, const Tensor& live) {
    if (!serialized.is_object() || serialized.size() != 4 ||
        serialized.at("version").get<string>() != "1.0.0" ||
        serialized.at("id").get<uint64_t>() != live.getOriginalId() ||
        serialized.at("data_type").get<DataType>() != live.getDataType() ||
        serialized.at("dimensions").get<vector<uint64_t>>() != live.getDimensions()) {
        throw runtime_error("RaggedLossShaper neutral tensor reference disagrees with its logical tensor.");
    }
}

void validateCanonicalInputReference(const json& serialized, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(serialized) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(serialized, "RaggedLossShaper") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        serialized.at("ragged_rank").get<uint32_t>() != 1 ||
        serialized.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow()) {
        throw runtime_error("RaggedLossShaper neutral input reference has invalid semantic geometry.");
    }
    validateCanonicalTensorReference(serialized.at("values"), live.getValues());
    validateCanonicalTensorReference(serialized.at("offsets"), live.getRowPartitionToken());
}

uint64_t checkedAdd(uint64_t left, uint64_t right) {
    if (left > numeric_limits<uint64_t>::max() - right)
        throw overflow_error("RaggedLossShaper memory requirement overflows uint64_t.");
    return left + right;
}
}  // namespace

RaggedLossShaper RaggedLossShaper::Builder::build() {
    if (!_network.has_value() || !_lossInput.has_value() || !_outputLossType.has_value())
        throw runtime_error("RaggedLossShaper requires network, ragged loss input, and an output loss type.");
    if (!_lossInput->isInitialized())
        throw invalid_argument("RaggedLossShaper input must be an initialized RaggedTensor.");
    if (_lossInput->getRaggedRank() != 1 || !_lossInput->hasMaxValuesPerRow() ||
        _lossInput->getOffsetsDimensions() != vector<uint64_t>{1} ||
        _lossInput->getValuesDimensions().front() != _lossInput->getMaxValuesPerRow())
        throw invalid_argument("RaggedLossShaper requires a logical rank-1 input with positive M.");
    if (_lossInput->getValuesDataType() != DataType::FP16 && _lossInput->getValuesDataType() != DataType::FP32)
        throw invalid_argument("RaggedLossShaper raw loss values must use FP16 or FP32.");
    if (_outputLossType.value() != ThorImplementation::RaggedLossShaper::OutputLossType::BATCH &&
        _outputLossType.value() != ThorImplementation::RaggedLossShaper::OutputLossType::PER_EXAMPLE) {
        throw invalid_argument("RaggedLossShaper API supports only BATCH and PER_EXAMPLE materialized outputs.");
    }

    RaggedLossShaper layer;
    layer.raggedLossInput = _lossInput.value();
    layer.outputLossType = _outputLossType.value();
    layer.lossOutput = Tensor(layer.raggedLossInput.getValuesDataType(), {1});
    layer.featureInputs = {layer.raggedLossInput.getValues(), layer.raggedLossInput.getOffsets()};
    layer.featureOutputs = {layer.lossOutput};
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

int RaggedLossShaper::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == raggedLossInput.getValues())
        return static_cast<int>(ThorImplementation::RaggedLossShaper::InputConnection::VALUES);
    if (connectingTensor == raggedLossInput.getOffsets())
        return static_cast<int>(ThorImplementation::RaggedLossShaper::InputConnection::OFFSETS);
    if (connectingTensor == lossOutput)
        return 0;
    throw runtime_error("Tensor is not connected to this RaggedLossShaper layer.");
}

vector<Tensor> RaggedLossShaper::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedOutput || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedOutput = true;
    return {lossOutput};
}

void RaggedLossShaper::informThatInputConnectionMade(Tensor inputTensor) {
    const int connectionType = getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(connectionType >= 0);
    connectedInputPortIndices.insert(static_cast<uint32_t>(connectionType));
}

void RaggedLossShaper::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedOutput = false;
}

optional<string> RaggedLossShaper::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedLossInput.getValues()) return "raw_loss_values";
    if (inputTensor == raggedLossInput.getOffsets()) return "offsets";
    return nullopt;
}

optional<string> RaggedLossShaper::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor != lossOutput) return nullopt;
    return outputLossType == ThorImplementation::RaggedLossShaper::OutputLossType::BATCH ? optional<string>("batch_loss")
                                                                                       : optional<string>("per_example_loss");
}

uint64_t RaggedLossShaper::getOutputTensorBytes(uint32_t batchSize) const {
    const uint64_t rows = outputLossType == ThorImplementation::RaggedLossShaper::OutputLossType::BATCH
        ? 1 : static_cast<uint64_t>(batchSize);
    const uint64_t bytesPerRow = lossOutput.getTotalSizeInBytes();
    if (rows == 0 || bytesPerRow > numeric_limits<uint64_t>::max() / rows)
        throw overflow_error("RaggedLossShaper output has zero batch or overflows uint64_t.");
    if (batchSize == 0)
        throw invalid_argument("RaggedLossShaper placement batch size must be nonzero.");
    return bytesPerRow * rows;
}

uint64_t RaggedLossShaper::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    uint64_t bytes = getOutputTensorBytes(batchSize);
    if (outputLossType != ThorImplementation::RaggedLossShaper::OutputLossType::BATCH) return bytes;

    const uint64_t b = static_cast<uint64_t>(batchSize);
    const DataType dtype = raggedLossInput.getValuesDataType();
    ThorImplementation::TensorDescriptor perExampleDescriptor(dtype, {b, 1});
    bytes = checkedAdd(bytes, perExampleDescriptor.getArraySizeInBytes());
    ThorImplementation::CubReduction reduction(ThorImplementation::CubReductionOp::Sum, {0}, dtype, 1.0f);
    Stream queryStream = Stream::getNextUploadStream(tensorPlacement.getDeviceNum());
    bytes = checkedAdd(bytes, reduction.queryWorkspaceSizeInBytes(perExampleDescriptor, queryStream));
    return bytes;
}

shared_ptr<ThorImplementation::Layer> RaggedLossShaper::stamp(ThorImplementation::TensorPlacement placement,
                                                               shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                               shared_ptr<Thor::Layer> drivingApiLayer,
                                                               Thor::Tensor connectingApiTensor,
                                                               bool inferenceOnly) const {
    (void)placement;
    (void)drivingApiLayer;
    (void)inferenceOnly;
    const int port = getConnectionType(connectingApiTensor);
    if (port != 0 && port != 1)
        throw runtime_error("RaggedLossShaper stamp requires an input values/offsets port.");
    // DEVICE_OFFSETS is an actual physical requirement of this reducer,
    // never a serialized model-batch property.
    ThorImplementation::RaggedTensorDescriptor physical;
    if (port == 0) {
        physical = SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            raggedLossInput, drivingLayer, "RaggedLossShaper");
    } else {
        const auto actualOffsets = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
            drivingLayer, "RaggedLossShaper");
        const auto dims = actualOffsets.getDimensions();
        if (actualOffsets.getDataType() != raggedLossInput.getOffsetsDataType() ||
            dims.size() != 1 || dims.front() < 2)
            throw runtime_error("RaggedLossShaper requires a real [B+1] offsets driver.");
        physical = raggedLossInput.getDescriptorForPlacement(dims.front() - 1);
        if (actualOffsets != physical.getOffsetsDescriptor())
            throw runtime_error("RaggedLossShaper offsets driver disagrees with runtime placement.");
        const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
        if (managed.has_value() && managed.value() != physical.getRowPartition())
            throw runtime_error("RaggedLossShaper managed offsets metadata disagrees with placement.");
    }
    const auto actualDriver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
        drivingLayer, "RaggedLossShaper");
    if (actualDriver != (port == 0 ? physical.getValuesDescriptor() : physical.getOffsetsDescriptor()))
        throw runtime_error("RaggedLossShaper physical driver descriptor disagrees with placement.");
    const uint64_t b = physical.getBatchSize();
    const uint64_t capacity = physical.getValuesDescriptor().getDimensions().front();
    if (b == 0 || b > numeric_limits<uint32_t>::max() || capacity == 0)
        throw runtime_error("RaggedLossShaper runtime batch or values capacity is invalid.");
    auto physicalLayer = make_shared<ThorImplementation::RaggedLossShaper>(outputLossType, b, capacity);
    physicalLayer->setName(getLayerType());
    return physicalLayer;
}

json RaggedLossShaper::architectureJson() const {
    return json{{"factory", Layer::Factory::Loss.value()},
                {"version", "1.1.0"},
                {"layer_type", "ragged_loss_shaper"},
                {"loss_shape", outputLossType == ThorImplementation::RaggedLossShaper::OutputLossType::BATCH
                                   ? Loss::LossShape::BATCH
                                   : Loss::LossShape::PER_EXAMPLE},
                {"ragged_loss_input", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedLossInput)},
                {"loss_output", lossOutput.architectureJson()}};
}

void RaggedLossShaper::deserialize(const json& j, Network* network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in RaggedLossShaper::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "ragged_loss_shaper")
        throw runtime_error("Layer type mismatch in RaggedLossShaper::deserialize.");
    if (version == "1.1.0" && (!j.is_object() || j.size() != 6 ||
                               j.at("factory") != Layer::Factory::Loss.value()))
        throw runtime_error("RaggedLossShaper neutral layer schema contains missing or extra fields.");

    const json& inputJson = j.at("ragged_loss_input");
    RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(inputJson, network, "RaggedLossShaper");
    if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(inputJson, "RaggedLossShaper") ==
        SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL)
        validateCanonicalInputReference(inputJson, input);
    const Loss::LossShape serializedShape = j.at("loss_shape").get<Loss::LossShape>();
    ThorImplementation::RaggedLossShaper::OutputLossType outputType;
    if (serializedShape == Loss::LossShape::BATCH) {
        outputType = ThorImplementation::RaggedLossShaper::OutputLossType::BATCH;
    } else if (serializedShape == Loss::LossShape::PER_EXAMPLE) {
        outputType = ThorImplementation::RaggedLossShaper::OutputLossType::PER_EXAMPLE;
    } else {
        throw runtime_error("Serialized RaggedLossShaper must be BATCH or PER_EXAMPLE.");
    }
    const json& outputJson = j.at("loss_output");
    Tensor output = Tensor::deserialize(outputJson);
    if (version == "1.1.0") {
        validateCanonicalTensorReference(outputJson, output);
    }
    if (output.getOriginalId() == input.getValues().getOriginalId() ||
        output.getOriginalId() == input.getOffsets().getOriginalId())
        throw runtime_error("RaggedLossShaper output must have a distinct tensor identity.");
    if (output.getDimensions() != vector<uint64_t>{1} || output.getDataType() != input.getValuesDataType())
        throw runtime_error("Serialized RaggedLossShaper output must be scalar-feature [1] with raw loss dtype.");

    RaggedLossShaper layer;
    layer.raggedLossInput = input;
    layer.outputLossType = outputType;
    layer.lossOutput = output;
    layer.featureInputs = {input.getValues(), input.getOffsets()};
    layer.featureOutputs = {output};
    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Loss::register_layer("ragged_loss_shaper", &Thor::RaggedLossShaper::deserialize);
    return true;
}();
}  // namespace
