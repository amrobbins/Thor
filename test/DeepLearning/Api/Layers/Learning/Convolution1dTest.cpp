#include "DeepLearning/Api/Layers/Learning/Convolution1d.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;
using DataType = ThorImplementation::DataType;
using json = nlohmann::json;
using std::string;
using std::vector;

TEST(Convolution1dApi, DefaultsToValidPaddingAndOwnsRankThreeWeights) {
    Api::Network network("conv1dDefaults");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 16}).dataType(DataType::FP16).build();

    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(5)
                                  .filterWidth(3)
                                  .build();

    ASSERT_TRUE(conv.isInitialized());
    EXPECT_EQ(conv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{5, 14}));
    EXPECT_EQ(conv.getPaddingMode(), Api::Convolution1dPaddingMode::VALID);
    EXPECT_EQ(conv.getPaddingLeft(), 0u);
    EXPECT_EQ(conv.getPaddingRight(), 0u);

    const json arch = conv.architectureJson();
    EXPECT_EQ(arch.at("layer_type").get<string>(), "convolution_1d");
    EXPECT_EQ(arch.at("version").get<string>(), "2.0.0");
    EXPECT_EQ(arch.at("data_layout").get<string>(), "NCW");
    EXPECT_EQ(arch.at("padding_mode").get<string>(), "valid");
    EXPECT_EQ(arch.at("groups").get<uint32_t>(), 1u);
    EXPECT_EQ(arch.at("parameters").at("weights").at("shape"), json::array({5, 3, 3}));
}

TEST(Convolution1dApi, SameUpperAndCausalResolveStrideAndDilationAwarePadding) {
    Api::Network sameNetwork("conv1dSame");
    Api::NetworkInput sameInput =
        Api::NetworkInput::Builder().network(sameNetwork).name("input").dimensions({2, 8}).dataType(DataType::FP16).build();
    Api::Convolution1d same = Api::Convolution1d::Builder()
                                  .network(sameNetwork)
                                  .featureInput(sameInput.getFeatureOutput().value())
                                  .numOutputChannels(4)
                                  .filterWidth(4)
                                  .stride(2)
                                  .dilation(2)
                                  .samePadding()
                                  .noActivation()
                                  .build();
    EXPECT_EQ(same.getPaddingMode(), Api::Convolution1dPaddingMode::SAME_UPPER);
    EXPECT_EQ(same.getPaddingLeft(), 2u);
    EXPECT_EQ(same.getPaddingRight(), 3u);
    EXPECT_EQ(same.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 4}));

    Api::Network causalNetwork("conv1dCausal");
    Api::NetworkInput causalInput =
        Api::NetworkInput::Builder().network(causalNetwork).name("input").dimensions({2, 8}).dataType(DataType::FP16).build();
    Api::Convolution1d causal = Api::Convolution1d::Builder()
                                    .network(causalNetwork)
                                    .featureInput(causalInput.getFeatureOutput().value())
                                    .numOutputChannels(4)
                                    .filterWidth(4)
                                    .stride(2)
                                    .dilation(2)
                                    .causalPadding()
                                    .noActivation()
                                    .build();
    EXPECT_EQ(causal.getPaddingMode(), Api::Convolution1dPaddingMode::CAUSAL);
    EXPECT_EQ(causal.getPaddingLeft(), 6u);
    EXPECT_EQ(causal.getPaddingRight(), 0u);
    EXPECT_EQ(causal.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 4}));
}

TEST(Convolution1dApi, ExplicitPaddingUsesIndependentLeftAndRight) {
    Api::Network network("conv1dExplicit");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({2, 11}).dataType(DataType::FP16).build();
    Api::Convolution1d conv = Api::Convolution1d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(3)
                                  .filterWidth(3)
                                  .stride(2)
                                  .dilation(2)
                                  .padding(2, 1)
                                  .noActivation()
                                  .build();
    EXPECT_EQ(conv.getPaddingMode(), Api::Convolution1dPaddingMode::EXPLICIT);
    EXPECT_EQ(conv.getPaddingLeft(), 2u);
    EXPECT_EQ(conv.getPaddingRight(), 1u);
    EXPECT_EQ(conv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{3, 5}));
    const json arch = conv.architectureJson();
    EXPECT_EQ(arch.at("padding_mode").get<string>(), "explicit");
    EXPECT_EQ(arch.at("padding_left").get<uint32_t>(), 2u);
    EXPECT_EQ(arch.at("padding_right").get<uint32_t>(), 1u);
}


TEST(Convolution1dApi, GroupedAndDepthwiseWeightsUsePerGroupInputChannels) {
    Api::Network groupedNetwork("conv1dGrouped");
    Api::NetworkInput groupedInput = Api::NetworkInput::Builder()
                                         .network(groupedNetwork)
                                         .name("input")
                                         .dimensions({8, 16})
                                         .dataType(DataType::FP16)
                                         .build();
    Api::Convolution1d grouped = Api::Convolution1d::Builder()
                                     .network(groupedNetwork)
                                     .featureInput(groupedInput.getFeatureOutput().value())
                                     .numOutputChannels(12)
                                     .filterWidth(3)
                                     .groups(4)
                                     .noActivation()
                                     .build();
    EXPECT_EQ(grouped.getGroups(), 4u);
    const json groupedArch = grouped.architectureJson();
    EXPECT_EQ(groupedArch.at("groups").get<uint32_t>(), 4u);
    EXPECT_EQ(groupedArch.at("parameters").at("weights").at("shape"), json::array({12, 2, 3}));

    Api::Network depthwiseNetwork("conv1dDepthwise");
    Api::NetworkInput depthwiseInput = Api::NetworkInput::Builder()
                                           .network(depthwiseNetwork)
                                           .name("input")
                                           .dimensions({8, 16})
                                           .dataType(DataType::FP16)
                                           .build();
    Api::Convolution1d depthwise = Api::Convolution1d::Builder()
                                       .network(depthwiseNetwork)
                                       .featureInput(depthwiseInput.getFeatureOutput().value())
                                       .numOutputChannels(8)
                                       .filterWidth(5)
                                       .groups(8)
                                       .causalPadding()
                                       .noActivation()
                                       .build();
    EXPECT_EQ(depthwise.architectureJson().at("parameters").at("weights").at("shape"), json::array({8, 1, 5}));
}

TEST(Convolution1dApi, RejectsInvalidGroupDivisibility) {
    Api::Network network("conv1dBadGroups");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({6, 16}).dataType(DataType::FP16).build();
    EXPECT_THROW((void)Api::Convolution1d::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .numOutputChannels(8)
                     .filterWidth(3)
                     .groups(4)
                     .build(),
                 std::invalid_argument);
}
