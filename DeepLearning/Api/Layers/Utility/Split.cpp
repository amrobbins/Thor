#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Split.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

Split::Split() = default;
Split::~Split() = default;

json Split::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInputs.size() == (getUseRagged() ? 2U : 1U));
    THOR_THROW_IF_FALSE(featureOutputs.size() == sizes.size());

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["axis"] = axis;
    j["sizes"] = sizes;
    if (getUseRagged()) {
        // The pieces share the input's partition.
        j["use_ragged"] = true;
        j["ragged_input"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput.value());
        json raggedOutputs = json::array();
        for (const RaggedTensor &output : raggedFeatureOutputs)
            raggedOutputs.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(output));
        j["ragged_outputs"] = raggedOutputs;
    } else {
        j["inputs"] = json::array({featureInputs[0].architectureJson()});
    }
    json outputs = json::array();
    for (const Tensor &output : featureOutputs) outputs.push_back(output.architectureJson());
    j["outputs"] = outputs;
    return j;
}

void Split::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0") throw runtime_error("Unsupported version in Split::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "split")
        throw runtime_error("Layer type mismatch in Split::deserialize: " + j.at("layer_type").get<string>());

    Split split;
    split.axis = j.at("axis").get<uint32_t>();
    split.sizes = j.at("sizes").get<vector<uint64_t>>();
    const bool ragged = j.value("use_ragged", false);
    const json &outputs = j.at("outputs");
    if (!outputs.is_array() || outputs.size() != split.sizes.size() || split.sizes.size() < 2)
        throw runtime_error("Split artifact needs one output per size, two or more.");

    // A ragged input's values carry the packed values dimension first, so the values axis is one further on.
    Tensor input;
    const json *raggedOutputs = nullptr;
    if (ragged) {
        raggedOutputs = &j.at("ragged_outputs");
        if (j.contains("inputs") || !raggedOutputs->is_array() || raggedOutputs->size() != split.sizes.size())
            throw runtime_error("Split ragged artifact needs a ragged input and one ragged output per size.");
        const json &raggedInput = j.at("ragged_input");
        if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(raggedInput, "Split") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL)
            throw runtime_error("Split ragged artifact requires a canonical ragged input reference.");
        split.raggedFeatureInput = SegmentedPrimitiveDetail::reconstructInput(raggedInput, network, "Split");
        input = split.raggedFeatureInput->getValues();
        split.featureInputs = {input, split.raggedFeatureInput->getOffsets()};
    } else {
        const json &inputs = j.at("inputs");
        if (j.contains("ragged_input") || j.contains("ragged_outputs") || !inputs.is_array() || inputs.size() != 1)
            throw runtime_error("Split artifact needs one input.");
        input = network->getApiTensorByOriginalId(inputs.at(0).at("id").get<uint64_t>());
        split.featureInputs.push_back(input);
    }
    const vector<uint64_t> dimensions = input.getDimensions();
    const uint32_t valuesAxis = split.axis + (ragged ? 1 : 0);
    if (valuesAxis >= dimensions.size()) throw runtime_error("Split artifact axis out of range.");

    // Each piece is the input's dimensions with its size on the axis; its type is its own.
    uint64_t total = 0;
    for (size_t i = 0; i < split.sizes.size(); ++i) {
        const Tensor piece =
            ragged ? SegmentedPrimitiveDetail::reconstructPreservedOutputValues(outputs.at(i), raggedOutputs->at(i),
                                                                                split.raggedFeatureInput.value(), "Split")
                   : Tensor::deserialize(outputs.at(i));
        vector<uint64_t> expected = dimensions;
        expected[valuesAxis] = split.sizes[i];
        if (split.sizes[i] == 0 || piece.getDimensions() != expected)
            throw runtime_error("Split artifact output " + to_string(i) + " does not match its size on the axis.");
        total += split.sizes[i];
        split.featureOutputs.push_back(piece);
        split.inputTensorFromOutputTensor[piece] = input;
        if (ragged) {
            if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(raggedOutputs->at(i), "Split") !=
                SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL)
                throw runtime_error("Split ragged artifact requires canonical ragged output references.");
            SegmentedPrimitiveDetail::validateSerializedPreservedPartition(raggedOutputs->at(i), j.at("ragged_input"),
                                                                           split.raggedFeatureInput.value(), "Split");
            split.raggedFeatureOutputs.push_back(split.raggedFeatureInput->withValues(piece));
        }
    }
    if (total != dimensions[valuesAxis]) throw runtime_error("Split artifact sizes do not sum to the input's extent.");

    split.initialized = true;
    split.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("split", &Thor::Split::deserialize);
    return true;
}();
}  // namespace
