#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;

namespace {

struct SimpleReluNetwork {
    Network network;
    NetworkInput input;
    std::shared_ptr<Activation> relu;
    NetworkOutput output;

    SimpleReluNetwork()
        : network("source_relu_network"),
          input(NetworkInput::Builder().network(network).name("features").dimensions({3}).dataType(DataType::FP32).build()),
          relu(Relu::Builder().network(network).featureInput(input.getFeatureOutput().value()).build()),
          output(NetworkOutput::Builder().network(network).name("scores").inputTensor(relu->getFeatureOutput().value()).build()) {}
};

}  // namespace

TEST(ApiSubgraphClone, ClonesInferenceSubgraphWithInputRemap) {
    SimpleReluNetwork source;

    Network destination("destination_relu_clone");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value());

    ApiSubgraphCloneOptions options;
    options.namePrefix = "member_0/";
    ApiSubgraphCloneResult cloneResult = destination.cloneInferenceSubgraphInto(source.network, {"scores"}, remap, options);

    ASSERT_EQ(cloneResult.outputTensorsByName.size(), 1u);
    ASSERT_TRUE(cloneResult.outputTensorsByName.count("scores"));
    Tensor clonedScores = cloneResult.outputTensorsByName.at("scores");
    EXPECT_TRUE(clonedScores.isInitialized());
    EXPECT_EQ(clonedScores.getDimensions(), std::vector<uint64_t>({3}));
    EXPECT_EQ(clonedScores.getDataType(), DataType::FP32);
    EXPECT_NE(clonedScores.getOriginalId(), source.output.getFeatureInput().value().getOriginalId());

    NetworkOutput::Builder().network(destination).name("scores").inputTensor(clonedScores).build();

    std::vector<std::string> inputNames = destination.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "features");
}

TEST(ApiSubgraphClone, TwoClonesOfSameSubgraphCanCoexist) {
    SimpleReluNetwork source;

    Network destination("destination_two_relu_clones");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value());

    ApiSubgraphCloneOptions options0;
    options0.namePrefix = "member_0/";
    ApiSubgraphCloneResult clone0 = destination.cloneInferenceSubgraphInto(source.network, {"scores"}, remap, options0);

    ApiSubgraphCloneOptions options1;
    options1.namePrefix = "member_1/";
    ApiSubgraphCloneResult clone1 = destination.cloneInferenceSubgraphInto(source.network, {"scores"}, remap, options1);

    Tensor scores0 = clone0.outputTensorsByName.at("scores");
    Tensor scores1 = clone1.outputTensorsByName.at("scores");
    EXPECT_TRUE(scores0.isInitialized());
    EXPECT_TRUE(scores1.isInitialized());
    EXPECT_NE(scores0, scores1);
    EXPECT_NE(scores0.getOriginalId(), scores1.getOriginalId());

    NetworkOutput::Builder().network(destination).name("scores_0").inputTensor(scores0).build();
    NetworkOutput::Builder().network(destination).name("scores_1").inputTensor(scores1).build();

    std::vector<std::string> inputNames = destination.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "features");
}

TEST(ApiSubgraphClone, ThrowsWhenRequiredInputIsNotRemapped) {
    SimpleReluNetwork source;
    Network destination("destination_missing_remap");

    ApiTensorRemap emptyRemap;
    EXPECT_THROW((destination.cloneInferenceSubgraphInto(source.network, {"scores"}, emptyRemap)), std::runtime_error);
}

TEST(ApiSubgraphClone, ThrowsWhenRequestedOutputIsMissing) {
    SimpleReluNetwork source;
    Network destination("destination_missing_output");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value());

    EXPECT_THROW((destination.cloneInferenceSubgraphInto(source.network, {"missing"}, remap)), std::runtime_error);
}

TEST(ApiSubgraphClone, ThrowsWhenRemapDescriptorsDoNotMatch) {
    SimpleReluNetwork source;
    Network destination("destination_bad_remap");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({4})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    EXPECT_THROW(remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value()), std::runtime_error);
}
