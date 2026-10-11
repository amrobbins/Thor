#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;

namespace {

std::shared_ptr<Network> makeReluPhase(const std::string& networkName,
                                       const std::string& inputName,
                                       const std::string& outputName,
                                       const std::vector<uint64_t>& dimensions = {3},
                                       DataType dataType = DataType::FP32,
                                       bool inputExternal = true,
                                       bool outputExternal = true) {
    auto network = std::make_shared<Network>(networkName);
    NetworkInput input = NetworkInput::Builder()
                             .network(*network)
                             .name(inputName)
                             .dimensions(dimensions)
                             .dataType(dataType)
                             .external(inputExternal)
                             .build();
    std::shared_ptr<Activation> relu = Relu::Builder().network(*network).featureInput(input.getFeatureOutput().value()).build();
    NetworkOutput::Builder()
        .network(*network)
        .name(outputName)
        .inputTensor(relu->getFeatureOutput().value())
        .external(outputExternal)
        .build();
    return network;
}

bool containsName(const std::vector<std::string>& names, const std::string& expected) {
    return std::find(names.begin(), names.end(), expected) != names.end();
}

}  // namespace

TEST(PhaseGraphConnector, ConnectsActivePhaseOutputToMatchingDownstreamInputByName) {
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_encoder", "features", "hidden");
    std::shared_ptr<Network> head = makeReluPhase("phase_graph_head", "hidden", "scores");

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName({{"encoder", encoder, true}, {"head", head, true}},
                                                             PhaseGraphComposeOptions{"phase_graph_encoder_head"});

    ASSERT_NE(graph.network, nullptr);
    ASSERT_EQ(graph.activePhaseNames.size(), 2u);
    EXPECT_EQ(graph.activePhaseNames[0], "encoder");
    EXPECT_EQ(graph.activePhaseNames[1], "head");

    ASSERT_EQ(graph.externalInputTensorsByName.size(), 1u);
    ASSERT_TRUE(graph.externalInputTensorsByName.count("features"));
    EXPECT_TRUE(graph.outputTensorsByName.count("hidden"));
    EXPECT_TRUE(graph.outputTensorsByName.count("scores"));

    std::vector<std::string> inputNames = graph.network->getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "features");
}

TEST(PhaseGraphConnector, SkipsInactivePhases) {
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_skip_encoder", "features", "hidden");
    std::shared_ptr<Network> disabledHead = makeReluPhase("phase_graph_skip_head", "hidden", "scores");

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName({{"encoder", encoder, true}, {"head", disabledHead, false}},
                                                             PhaseGraphComposeOptions{"phase_graph_skip_disabled"});

    ASSERT_EQ(graph.activePhaseNames.size(), 1u);
    EXPECT_EQ(graph.activePhaseNames[0], "encoder");
    ASSERT_EQ(graph.externalInputTensorsByName.size(), 1u);
    EXPECT_TRUE(graph.externalInputTensorsByName.count("features"));
    EXPECT_TRUE(graph.outputTensorsByName.count("hidden"));
    EXPECT_FALSE(graph.outputTensorsByName.count("scores"));
}

TEST(PhaseGraphConnector, DuplicateActiveOutputNamesThrow) {
    std::shared_ptr<Network> phase0 = makeReluPhase("phase_graph_duplicate_0", "features", "hidden");
    std::shared_ptr<Network> phase1 = makeReluPhase("phase_graph_duplicate_1", "other_features", "hidden");

    EXPECT_THROW((buildComposedPhaseGraphByName({{"phase0", phase0, true}, {"phase1", phase1, true}},
                                                PhaseGraphComposeOptions{"phase_graph_duplicate_outputs"})),
                 std::runtime_error);
}

TEST(PhaseGraphConnector, ScansAheadAndConnectsLaterProducerBeforeEarlierConsumer) {
    std::shared_ptr<Network> head = makeReluPhase("phase_graph_late_head", "hidden", "scores");
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_late_encoder", "features", "hidden");

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName({{"head", head, true}, {"encoder", encoder, true}},
                                                             PhaseGraphComposeOptions{"phase_graph_late_producer"});

    ASSERT_EQ(graph.activePhaseNames.size(), 2u);
    EXPECT_EQ(graph.activePhaseNames[0], "encoder");
    EXPECT_EQ(graph.activePhaseNames[1], "head");
    ASSERT_EQ(graph.externalInputTensorsByName.size(), 1u);
    EXPECT_TRUE(graph.externalInputTensorsByName.count("features"));
    EXPECT_FALSE(graph.externalInputTensorsByName.count("hidden"));
    EXPECT_TRUE(graph.outputTensorsByName.count("hidden"));
    EXPECT_TRUE(graph.outputTensorsByName.count("scores"));
}


TEST(PhaseGraphConnector, NonExternalInputWithoutActiveProducerThrows) {
    std::shared_ptr<Network> head = makeReluPhase("phase_graph_internal_input_missing",
                                                  "hidden",
                                                  "scores",
                                                  {3},
                                                  DataType::FP32,
                                                  /*inputExternal=*/false,
                                                  /*outputExternal=*/true);

    EXPECT_THROW((buildComposedPhaseGraphByName({{"head", head, true}}, PhaseGraphComposeOptions{"phase_graph_missing_internal"})),
                 std::runtime_error);
}

TEST(PhaseGraphConnector, NonExternalInputCanBeSatisfiedByActiveProducer) {
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_internal_input_encoder", "features", "hidden");
    std::shared_ptr<Network> head = makeReluPhase("phase_graph_internal_input_head",
                                                  "hidden",
                                                  "scores",
                                                  {3},
                                                  DataType::FP32,
                                                  /*inputExternal=*/false,
                                                  /*outputExternal=*/true);

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName({{"head", head, true}, {"encoder", encoder, true}},
                                                             PhaseGraphComposeOptions{"phase_graph_internal_satisfied"});

    ASSERT_EQ(graph.activePhaseNames.size(), 2u);
    EXPECT_EQ(graph.activePhaseNames[0], "encoder");
    EXPECT_EQ(graph.activePhaseNames[1], "head");
    ASSERT_EQ(graph.externalInputTensorsByName.size(), 1u);
    EXPECT_TRUE(graph.externalInputTensorsByName.count("features"));
    EXPECT_FALSE(graph.externalInputTensorsByName.count("hidden"));
}

TEST(PhaseGraphConnector, NonExternalOutputFeedsLocalConsumerButIsNotMaterialized) {
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_internal_output_encoder",
                                                     "features",
                                                     "hidden",
                                                     {3},
                                                     DataType::FP32,
                                                     /*inputExternal=*/true,
                                                     /*outputExternal=*/false);
    std::shared_ptr<Network> head = makeReluPhase("phase_graph_internal_output_head", "hidden", "scores");

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName({{"encoder", encoder, true}, {"head", head, true}},
                                                             PhaseGraphComposeOptions{"phase_graph_internal_output"});

    EXPECT_TRUE(graph.outputTensorsByName.count("hidden"));
    EXPECT_TRUE(graph.outputTensorsByName.count("scores"));

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = graph.network->place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    EXPECT_FALSE(containsName(placed->getStampedNetwork(0).getNamedOutputNames(), "hidden"));
    EXPECT_TRUE(containsName(placed->getStampedNetwork(0).getNamedOutputNames(), "scores"));
}

TEST(PhaseGraphConnector, MatchingOutputInputDescriptorMismatchThrows) {
    std::shared_ptr<Network> phase0 = makeReluPhase("phase_graph_mismatch_0", "features", "hidden", {3}, DataType::FP32);
    std::shared_ptr<Network> phase1 = makeReluPhase("phase_graph_mismatch_1", "hidden", "scores", {4}, DataType::FP32);

    EXPECT_THROW((buildComposedPhaseGraphByName({{"phase0", phase0, true}, {"phase1", phase1, true}},
                                                PhaseGraphComposeOptions{"phase_graph_descriptor_mismatch"})),
                 std::runtime_error);
}

TEST(PhaseGraphConnector, OnePhaseOutputCanFeedMultipleDownstreamConsumersThroughTensorFanout) {
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_fanout_encoder", "features", "hidden");
    std::shared_ptr<Network> head0 = makeReluPhase("phase_graph_fanout_head0", "hidden", "scores0");
    std::shared_ptr<Network> head1 = makeReluPhase("phase_graph_fanout_head1", "hidden", "scores1");

    ComposedPhaseGraph graph = buildComposedPhaseGraphByName(
        {{"encoder", encoder, true}, {"head0", head0, true}, {"head1", head1, true}},
        PhaseGraphComposeOptions{"phase_graph_fanout"});

    ASSERT_TRUE(graph.outputTensorsByName.count("hidden"));
    ASSERT_TRUE(graph.outputTensorsByName.count("scores0"));
    ASSERT_TRUE(graph.outputTensorsByName.count("scores1"));

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = graph.network->place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    bool foundTensorFanout = false;
    for (const std::shared_ptr<ThorImplementation::Layer>& layer : placed->getStampedNetwork(0).getOtherLayers()) {
        if (std::dynamic_pointer_cast<ThorImplementation::TensorFanout>(layer) != nullptr) {
            foundTensorFanout = true;
            break;
        }
    }
    EXPECT_TRUE(foundTensorFanout);

    std::vector<std::string> inputNames = placed->getNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "features");
    EXPECT_TRUE(containsName(placed->getStampedNetwork(0).getNamedOutputNames(), "hidden"));
    EXPECT_TRUE(containsName(placed->getStampedNetwork(0).getNamedOutputNames(), "scores0"));
    EXPECT_TRUE(containsName(placed->getStampedNetwork(0).getNamedOutputNames(), "scores1"));
}

TEST(PhaseGraphConnector, RecompositionPreservesCloneSourceKeysForMatchingPhaseLayers) {
    std::shared_ptr<Network> encoder = makeReluPhase("phase_graph_state_encoder", "features", "hidden");
    std::shared_ptr<Network> head = makeReluPhase("phase_graph_state_head", "hidden", "scores");

    ComposedPhaseGraph first = buildComposedPhaseGraphByName({{"encoder", encoder, true}},
                                                             PhaseGraphComposeOptions{"phase_graph_state_first"});
    ComposedPhaseGraph second = buildComposedPhaseGraphByName({{"encoder", encoder, true}, {"head", head, true}},
                                                              PhaseGraphComposeOptions{"phase_graph_state_second"});

    std::set<std::string> firstKeys;
    for (const auto& [layerId, key] : first.network->getCloneSourceKeysByLayerId()) {
        (void)layerId;
        firstKeys.insert(key);
    }

    std::set<std::string> secondKeys;
    for (const auto& [layerId, key] : second.network->getCloneSourceKeysByLayerId()) {
        (void)layerId;
        secondKeys.insert(key);
    }

    ASSERT_FALSE(firstKeys.empty());
    for (const std::string& key : firstKeys) {
        EXPECT_TRUE(secondKeys.count(key)) << key;
    }
}

TEST(PhaseGraphConnector, ComposedRaggedBoundaryKeepsSemanticMAndTwoPlacementCapacities) {
    auto makePhase = [](const std::string& name, const std::string& outputName) {
        auto network = std::make_shared<Network>(name);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(*network)
                                 .name("history")
                                 .valuesDataType(DataType::FP32)
                                 .trailingDimensions({2})
                                 .maxValuesPerRow(3)
                                 .build();
        std::shared_ptr<Activation> relu = Relu::Builder().network(*network).featureInput(input).build();
        if (!relu->getRaggedFeatureOutput().has_value()) {
            throw std::runtime_error("ragged phase Relu must preserve its logical ragged output");
        }
        NetworkOutput::Builder()
            .network(*network)
            .name(outputName)
            .inputTensor(relu->getRaggedFeatureOutput()->getValues())
            .build();
        return network;
    };

    const auto first = makePhase("ragged_phase_one", "first_output");
    const auto second = makePhase("ragged_phase_two", "second_output");
    ComposedPhaseGraph composed = buildComposedPhaseGraphByName(
        {{"first", first, true}, {"second", second, true}}, PhaseGraphComposeOptions{"ragged_composed"});
    ASSERT_NE(composed.network, nullptr);
    const auto inputs = composed.network->getExternalRaggedNetworkInputs();
    ASSERT_EQ(inputs.size(), 1u);
    EXPECT_EQ(inputs.front().raggedTensor.getValuesDimensions(), (std::vector<uint64_t>{3, 2}));
    EXPECT_EQ(inputs.front().raggedTensor.getOffsetsDimensions(), (std::vector<uint64_t>{1}));
    EXPECT_EQ(inputs.front().raggedTensor.getDescriptorForPlacement(2).getMaxTotalValues(), 6u);
    EXPECT_EQ(inputs.front().raggedTensor.getDescriptorForPlacement(7).getMaxTotalValues(), 21u);
}
