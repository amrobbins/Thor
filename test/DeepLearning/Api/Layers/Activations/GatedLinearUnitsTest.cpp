#include "DeepLearning/Api/Layers/Activations/BilinearGlu.h"
#include "DeepLearning/Api/Layers/Activations/Geglu.h"
#include "DeepLearning/Api/Layers/Activations/Glu.h"
#include "DeepLearning/Api/Layers/Activations/Reglu.h"
#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <memory>
#include <vector>

using namespace Thor;
using namespace std;

namespace {

template <typename ActivationT>
shared_ptr<ActivationT> buildActivation(Network& network, Tensor input) {
    typename ActivationT::Builder builder;
    shared_ptr<Activation> base = builder.network(network).featureInput(input).build();
    shared_ptr<ActivationT> activation = dynamic_pointer_cast<ActivationT>(base);
    EXPECT_NE(activation, nullptr);
    return activation;
}

template <typename ActivationT>
void expectGatedActivationBuilds(const string& expectedLayerType, const string& expectedSerializedType) {
    Network network("testNetwork");
    Tensor featureInput(DataType::FP32, {2, 3, 10});

    shared_ptr<ActivationT> activation = buildActivation<ActivationT>(network, featureInput);
    ASSERT_NE(activation, nullptr);
    ASSERT_TRUE(activation->isInitialized());
    ASSERT_EQ(activation->getLayerType(), expectedLayerType);

    ASSERT_TRUE(activation->getFeatureInput().has_value());
    ASSERT_EQ(activation->getFeatureInput().value(), featureInput);

    ASSERT_TRUE(activation->getFeatureOutput().has_value());
    ASSERT_EQ(activation->getFeatureOutput().value().getDataType(), DataType::FP32);
    ASSERT_EQ(activation->getFeatureOutput().value().getDimensions(), (vector<uint64_t>{2, 3, 5}));

    const auto json = activation->architectureJson();
    ASSERT_EQ(json.at("layer_type").template get<string>(), expectedSerializedType);

    ThorImplementation::Expression input = ThorImplementation::Expression::input("feature_input");
    EXPECT_NO_THROW((void)activation->toExpression(input));
}

}  // namespace

TEST(GatedLinearUnits, BuildHalvesFinalFeatureDimension) {
    expectGatedActivationBuilds<Glu>("Glu", "glu");
    expectGatedActivationBuilds<Reglu>("Reglu", "reglu");
    expectGatedActivationBuilds<Geglu>("Geglu", "geglu");
    expectGatedActivationBuilds<Swiglu>("Swiglu", "swiglu");
    expectGatedActivationBuilds<BilinearGlu>("BilinearGlu", "bilinear_glu");
}

TEST(GatedLinearUnits, RejectOddFinalFeatureDimension) {
    Network network("testNetwork");
    Tensor featureInput(DataType::FP32, {2, 3, 9});

    Glu::Builder builder;
    EXPECT_ANY_THROW((void)builder.network(network).featureInput(featureInput).build());
}



TEST(GatedLinearUnits, StandaloneFourArgumentAddToNetworkPreservesShapeChangingDispatch) {
    Network network("standaloneFourArgumentAddToNetworkPreservesShapeChangingDispatch");
    Tensor featureInput(DataType::BF16, {53, 512});

    // A directly default-constructed API layer is intentionally uninitialized and
    // may not be inserted into a network. Build a detached, initialized activation
    // (no feature input supplied to the builder), then exercise the four-argument
    // virtual dispatch path that the Python binding uses for standalone activations.
    shared_ptr<Activation> activation = Swiglu::Builder().build();
    ASSERT_NE(activation, nullptr);
    ASSERT_TRUE(activation->isInitialized());

    Tensor featureOutput = activation->addToNetwork(featureInput, &network, std::nullopt, {});

    EXPECT_EQ(featureOutput.getDataType(), DataType::BF16);
    EXPECT_EQ(featureOutput.getDimensions(), (vector<uint64_t>{53, 256}));
}

TEST(GatedLinearUnits, StandaloneActivationEpilogueIsRejectedExplicitly) {
    Network network("standaloneActivationEpilogueIsRejectedExplicitly");
    Tensor featureInput(DataType::BF16, {53, 512});

    shared_ptr<Activation> activation = Swiglu::Builder().build();
    ASSERT_NE(activation, nullptr);
    ASSERT_TRUE(activation->isInitialized());

    ThorImplementation::Expression epilogueInput = Activation::epilogueInput(DataType::FP32, DataType::BF16);
    EXPECT_THROW(
        (void)activation->addToNetwork(featureInput, &network, epilogueInput, {}),
        std::invalid_argument);
}

TEST(GatedLinearUnits, PlacedSwigluPreservesBatchAndPrefixDimensions) {
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t sequenceLength = 5;
    constexpr uint64_t outputWidth = 4;

    Network network("placedSwigluPreservesBatchAndPrefixDimensions");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({sequenceLength, 2 * outputWidth})
                             .dataType(DataType::BF16)
                             .build();
    shared_ptr<Swiglu> swiglu = buildActivation<Swiglu>(network, input.getFeatureOutput().value());
    ASSERT_NE(swiglu, nullptr);
    NetworkOutput output = NetworkOutput::Builder()
                               .network(network)
                               .name("output")
                               .inputTensor(swiglu->getFeatureOutput().value())
                               .dataType(DataType::BF16)
                               .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placedNetwork;
    ASSERT_NO_THROW(placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true));
    for (Event& event : initDoneEvents)
        event.synchronize();
    ASSERT_NE(placedNetwork, nullptr);

    ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);

    // Network::addToNetwork() stores a clone of each API activation, and activation
    // clone() methods intentionally assign the stored clone a fresh API-layer id.
    // Therefore the id on the builder-returned `swiglu` object is not the id used
    // by the stamped network. Locate the standalone physical activation by type
    // rather than querying with the pre-clone id.
    shared_ptr<ThorImplementation::CustomLayer> physicalSwiglu;
    for (const auto& [apiLayerId, physicalLayer] : stampedNetwork.getApiLayerToPhysicalLayer()) {
        (void)apiLayerId;
        auto candidate = dynamic_pointer_cast<ThorImplementation::CustomLayer>(physicalLayer);
        if (candidate != nullptr && candidate->getLayerType().starts_with("CustomLayer<Swiglu#")) {
            physicalSwiglu = candidate;
            break;
        }
    }

    auto physicalOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(
        stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalSwiglu, nullptr);
    ASSERT_TRUE(physicalSwiglu->getFeatureOutput().has_value());
    EXPECT_EQ(physicalSwiglu->getFeatureOutput()->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, outputWidth}));

    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_TRUE(physicalOutput->getFeatureOutput().has_value());
    EXPECT_EQ(physicalOutput->getFeatureOutput()->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, outputWidth}));
}

TEST(GatedLinearUnits, TransformerSwiGluFeedForwardBlockPlacesWithTokenwiseResidualProjection) {
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t sequenceLength = 53;
    constexpr uint64_t modelWidth = 128;
    constexpr uint64_t feedForwardWidth = 256;

    Network network("transformerSwiGluFeedForwardBlock");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({sequenceLength, modelWidth})
                             .dataType(DataType::BF16)
                             .build();

    RMSNorm normalized = RMSNorm::Builder()
                             .network(network)
                             .featureInput(input.getFeatureOutput().value())
                             .normalizedShape({modelWidth})
                             .parameterDataType(DataType::FP32)
                             .build();

    FullyConnected gateAndValue = FullyConnected::Builder()
                                      .network(network)
                                      .featureInput(normalized.getFeatureOutput().value())
                                      .numOutputFeatures(2 * feedForwardWidth)
                                      .preserveInputPrefixDimensions(true)
                                      .hasBias(true)
                                      .weightsDataType(DataType::BF16)
                                      .computeDataType(DataType::FP32)
                                      .outputDataType(DataType::BF16)
                                      .noActivation()
                                      .build();

    shared_ptr<Swiglu> swiglu = buildActivation<Swiglu>(network, gateAndValue.getFeatureOutput().value());
    ASSERT_NE(swiglu, nullptr);

    ThorImplementation::Expression projected = FullyConnected::epilogueInput(DataType::FP32, DataType::BF16);
    ThorImplementation::Expression residual =
        FullyConnected::epilogueAuxInput("residual", DataType::FP32, DataType::BF16);
    FullyConnected outputProjection = FullyConnected::Builder()
                                          .network(network)
                                          .featureInput(swiglu->getFeatureOutput().value())
                                          .numOutputFeatures(modelWidth)
                                          .preserveInputPrefixDimensions(true)
                                          .hasBias(true)
                                          .weightsDataType(DataType::BF16)
                                          .computeDataType(DataType::FP32)
                                          .outputDataType(DataType::BF16)
                                          .noActivation()
                                          .epilogueInput("residual", input.getFeatureOutput().value())
                                          .epilogue(projected + residual)
                                          .build();

    NetworkOutput output = NetworkOutput::Builder()
                               .network(network)
                               .name("output")
                               .inputTensor(outputProjection.getFeatureOutput().value())
                               .dataType(DataType::BF16)
                               .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placedNetwork;
    ASSERT_NO_THROW(placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true));
    for (Event& event : initDoneEvents)
        event.synchronize();
    ASSERT_NE(placedNetwork, nullptr);

    ThorImplementation::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(
        stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_TRUE(physicalOutput->getFeatureOutput().has_value());
    EXPECT_EQ(physicalOutput->getFeatureOutput()->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, modelWidth}));
}
