#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"

#include <limits>
#include <set>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {
// The saved original IDs, never deserialization-assigned live IDs, are the
// identity of referenced graph tensors in MODEL_RECONSTRUCTION.
void validateNeutralTensorReference(const json& serialized, const Tensor& live) {
    if (!serialized.is_object() || serialized.size() != 4 ||
        serialized.at("version").get<string>() != "1.0.0" ||
        serialized.at("id").get<uint64_t>() != live.getOriginalId() ||
        serialized.at("data_type").get<DataType>() != live.getDataType() ||
        serialized.at("dimensions").get<vector<uint64_t>>() != live.getDimensions()) {
        throw runtime_error("Concatenate neutral reference does not match canonical graph tensor.");
    }
}

void validateNeutralRaggedReference(const json& serialized, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(serialized) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(serialized, "Concatenate") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        serialized.at("ragged_rank").get<uint32_t>() != 1 ||
        serialized.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        serialized.contains("batch_size") || serialized.contains("max_total_values")) {
        throw runtime_error("Concatenate neutral reference has invalid semantic metadata.");
    }
    validateNeutralTensorReference(serialized.at("values"), live.getValues());
    validateNeutralTensorReference(serialized.at("offsets"), live.getRowPartitionToken());
}

// Validate geometry again during deserialize: a forged artifact can have
// individually valid input references yet an invalid output/axis relationship.
void validateNeutralConcatGeometry(const vector<RaggedTensor>& inputs, const Tensor& output,
                                   uint32_t axis) {
    if (inputs.size() < 2)
        throw runtime_error("Concatenate neutral artifact requires two or more neutral inputs.");
    const RaggedTensor& reference = inputs.front();
    const vector<uint64_t> trailing = reference.getTrailingDimensions();
    if (axis >= trailing.size()) throw runtime_error("Concatenate neutral artifact axis out of range.");
    vector<uint64_t> expected = reference.getValuesDimensions();
    expected.at(axis + 1) = 0;
    set<uint64_t> valueIds;
    for (const RaggedTensor& input : inputs) {
        if (!input.sharesPartitionWith(reference) ||
            input.getRowPartitionToken() != reference.getRowPartitionToken() ||
            input.getMaxValuesPerRow() != reference.getMaxValuesPerRow() ||
            input.getValuesDataType() != reference.getValuesDataType() ||
            input.getOffsetsDataType() != reference.getOffsetsDataType() ||
            input.getTrailingDimensions().size() != trailing.size() ||
            !valueIds.insert(input.getValues().getOriginalId()).second) {
            throw runtime_error("Concatenate neutral inputs disagree on partition, dtype, M or port identity.");
        }
        const auto dims = input.getTrailingDimensions();
        for (uint32_t d = 0; d < dims.size(); ++d) {
            if (d != axis && dims[d] != trailing[d])
                throw runtime_error("Concatenate neutral non-concatenated dimensions disagree.");
        }
        if (expected[axis + 1] > numeric_limits<uint64_t>::max() - dims[axis])
            throw overflow_error("Concatenate neutral output dimension overflows.");
        expected[axis + 1] += dims[axis];
    }
    if (output.getDataType() != reference.getValuesDataType() ||
        output.getDimensions() != expected || valueIds.count(output.getOriginalId())) {
        throw runtime_error("Concatenate neutral output does not match concatenated semantic dimensions.");
    }
}
} // namespace

Concatenate::Concatenate() = default;
Concatenate::~Concatenate() = default;

json Concatenate::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getUseRagged() ? "1.1.0" : getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["concatenation_axis"] = concatenationAxis;

    if (getUseRagged()) {
        j["use_ragged"] = true;
        json raggedInputs = json::array();
        for (const RaggedTensor &input : raggedFeatureInputs)
            raggedInputs.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(input));
        j["ragged_inputs"] = raggedInputs;
        j["ragged_output"] = SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput.value());
    } else {
        json inputs = json::array();
        for (const Tensor &input : featureInputs) inputs.push_back(input.architectureJson());
        j["inputs"] = inputs;
    }

    json outputs = json::array();
    for (const Tensor &output : featureOutputs) outputs.push_back(output.architectureJson());
    j["outputs"] = outputs;
    return j;
}

void Concatenate::deserialize(const json &j, Network *network) {
    const string version = j.at("version").get<string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw runtime_error("Unsupported version in Concatenate::deserialize: " + version);
    if (j.at("layer_type").get<string>() != "concatenate")
        throw runtime_error("Layer type mismatch in Concatenate::deserialize: " + j.at("layer_type").get<string>());

    Concatenate concatenate;
    concatenate.concatenationAxis = j.at("concatenation_axis").get<uint32_t>();
    concatenate.numInputConnectionsMade = 0;

    if (j.value("use_ragged", false)) {
        const bool neutral = version == "1.1.0";
        const json& inputJson = j.at("ragged_inputs");
        if (!inputJson.is_array() || inputJson.size() < 2 ||
            (neutral && (j.contains("inputs") || !j.at("outputs").is_array() ||
                         j.at("outputs").size() != 1))) {
            throw runtime_error("Concatenate malformed ragged input/output arrays.");
        }
        auto restoreRagged = [&](const json &r) {
            const auto grammar = SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(r, "Concatenate");
            if ((grammar == SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) != neutral)
                throw runtime_error("Concatenate layer version disagrees with serialized reference grammar.");
            RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(r, network, "Concatenate");
            if (neutral) validateNeutralRaggedReference(r, input);
            return input;
        };
        for (const json &input : inputJson)
            concatenate.raggedFeatureInputs.push_back(restoreRagged(input));
        THOR_THROW_IF_FALSE(concatenate.raggedFeatureInputs.size() >= 2);
        for (const RaggedTensor &input : concatenate.raggedFeatureInputs)
            concatenate.featureInputs.push_back(input.getValues());
        concatenate.featureInputs.push_back(concatenate.raggedFeatureInputs.front().getOffsets());

        Tensor outputValues = SegmentedPrimitiveDetail::reconstructPreservedOutputValues(
            j.at("outputs").at(0), j.at("ragged_output"),
            concatenate.raggedFeatureInputs.front(), "Concatenate");
        concatenate.featureOutputs = {outputValues};
        concatenate.raggedFeatureOutput = concatenate.raggedFeatureInputs.front().withValues(outputValues);
        const json &serializedOutput = j.at("ragged_output");
        if (serializedOutput.at("offsets").at("id").get<uint64_t>() !=
                j.at("ragged_inputs").at(0).at("offsets").at("id").get<uint64_t>() ||
            serializedOutput.at("values").at("id").get<uint64_t>() !=
                j.at("outputs").at(0).at("id").get<uint64_t>()) {
            throw runtime_error("Concatenate serialized ragged output does not preserve the shared row partition.");
        }
        // Geometry and partition sharing are semantic requirements for every
        // recognized artifact version, not merely for the newest JSON syntax.
        validateNeutralConcatGeometry(concatenate.raggedFeatureInputs, outputValues, concatenate.concatenationAxis);
        SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
            serializedOutput, inputJson.at(0), concatenate.raggedFeatureInputs.front(), "Concatenate");
        if (neutral) {
            validateNeutralRaggedReference(serializedOutput, concatenate.raggedFeatureOutput.value());
            // Each nested output tensor must be the same canonical output as the
            // standalone outputs[0], including saved original ID and geometry.
            validateNeutralTensorReference(j.at("outputs").at(0), outputValues);
        }
    } else {
        if (version != "1.0.0")
            throw runtime_error("Concatenate 1.1.0 is reserved for neutral ragged geometry.");
        for (const json &input : j.at("inputs")) {
            concatenate.featureInputs.push_back(
                network->getApiTensorByOriginalId(input.at("id").get<uint64_t>()));
        }
        THOR_THROW_IF_FALSE(concatenate.featureInputs.size() >= 2);
        concatenate.featureOutputs = {Tensor::deserialize(j.at("outputs").at(0))};
    }

    for (const Tensor &input : concatenate.featureInputs)
        concatenate.outputTensorFromInputTensor[input] = concatenate.featureOutputs[0];
    concatenate.initialized = true;
    concatenate.addToNetwork(network);
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("concatenate", &Thor::Concatenate::deserialize);
    return true;
}();
}  // namespace
