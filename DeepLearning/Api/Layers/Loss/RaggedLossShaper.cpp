#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace Thor {

RaggedLossShaper RaggedLossShaper::Builder::build() {
    if (!_network.has_value() || !_lossInput.has_value() || !_outputLossType.has_value())
        throw runtime_error("RaggedLossShaper requires network, ragged loss input, and an output loss type.");
    if (!_lossInput->isInitialized())
        throw invalid_argument("RaggedLossShaper input must be an initialized RaggedTensor.");
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
    (void)batchSize;
    const uint64_t rows = outputLossType == ThorImplementation::RaggedLossShaper::OutputLossType::BATCH
        ? 1
        : raggedLossInput.getBatchSize();
    return rows * lossOutput.getTotalSizeInBytes();
}

uint64_t RaggedLossShaper::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)batchSize;
    uint64_t bytes = getOutputTensorBytes(batchSize);
    if (outputLossType != ThorImplementation::RaggedLossShaper::OutputLossType::BATCH) return bytes;

    const uint64_t logicalBatchSize = raggedLossInput.getBatchSize();
    const DataType dtype = raggedLossInput.getValuesDataType();
    ThorImplementation::TensorDescriptor perExampleDescriptor(dtype, {logicalBatchSize, 1});
    bytes += perExampleDescriptor.getArraySizeInBytes();
    ThorImplementation::CubReduction reduction(ThorImplementation::CubReductionOp::Sum, {0}, dtype, 1.0f);
    Stream queryStream = Stream::getNextUploadStream(tensorPlacement.getDeviceNum());
    bytes += reduction.queryWorkspaceSizeInBytes(perExampleDescriptor, queryStream);
    return bytes;
}

shared_ptr<ThorImplementation::Layer> RaggedLossShaper::stamp(ThorImplementation::TensorPlacement placement,
                                                               shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                               shared_ptr<Thor::Layer> drivingApiLayer,
                                                               Thor::Tensor connectingApiTensor,
                                                               bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)inferenceOnly;
    (void)getConnectionType(connectingApiTensor);
    auto physical = make_shared<ThorImplementation::RaggedLossShaper>(
        outputLossType, raggedLossInput.getBatchSize(), raggedLossInput.getMaxTotalValues());
    physical->setName(getLayerType());
    return physical;
}

json RaggedLossShaper::architectureJson() const {
    return json{{"factory", Layer::Factory::Loss.value()},
                {"version", getLayerVersion()},
                {"layer_type", "ragged_loss_shaper"},
                {"loss_shape", outputLossType == ThorImplementation::RaggedLossShaper::OutputLossType::BATCH
                                   ? Loss::LossShape::BATCH
                                   : Loss::LossShape::PER_EXAMPLE},
                {"ragged_loss_input", raggedLossInput.architectureJson()},
                {"loss_output", lossOutput.architectureJson()}};
}

void RaggedLossShaper::deserialize(const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in RaggedLossShaper::deserialize: " + j.at("version").get<string>());
    if (j.at("layer_type").get<string>() != "ragged_loss_shaper")
        throw runtime_error("Layer type mismatch in RaggedLossShaper::deserialize: " + j.at("layer_type").get<string>());

    RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_loss_input"), network, "RaggedLossShaper");
    const Loss::LossShape serializedShape = j.at("loss_shape").get<Loss::LossShape>();
    ThorImplementation::RaggedLossShaper::OutputLossType outputType;
    if (serializedShape == Loss::LossShape::BATCH) {
        outputType = ThorImplementation::RaggedLossShaper::OutputLossType::BATCH;
    } else if (serializedShape == Loss::LossShape::PER_EXAMPLE) {
        outputType = ThorImplementation::RaggedLossShaper::OutputLossType::PER_EXAMPLE;
    } else {
        throw runtime_error("Serialized RaggedLossShaper must be BATCH or PER_EXAMPLE.");
    }
    Tensor output = Tensor::deserialize(j.at("loss_output"));
    if (output.getDimensions() != vector<uint64_t>{1} || output.getDataType() != input.getValuesDataType())
        throw runtime_error("Serialized RaggedLossShaper output must be scalar-feature [1] with the raw loss dtype.");

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
