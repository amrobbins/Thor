#include "DeepLearning/Api/Layers/Utility/FiniteCheck.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, FiniteCheckBuildsAsLogicalIdentity) {
    Network network("finiteCheckBuilds");
    Tensor featureInput(DataType::BF16, {3, 5});

    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(network)
                                  .featureInput(featureInput)
                                  .tensorLabel("after_projection")
                                  .checkForward(true)
                                  .checkBackward(false)
                                  .failOnNonFinite(false)
                                  .maxReportedIndices(4)
                                  .build();

    ASSERT_TRUE(finiteCheck.isInitialized());
    ASSERT_TRUE(finiteCheck.getFeatureInput().has_value());
    ASSERT_TRUE(finiteCheck.getFeatureOutput().has_value());
    EXPECT_EQ(finiteCheck.getFeatureInput().value(), featureInput);
    EXPECT_NE(finiteCheck.getFeatureOutput().value(), featureInput);
    EXPECT_EQ(finiteCheck.getFeatureOutput().value().getDataType(), DataType::BF16);
    EXPECT_EQ(finiteCheck.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{3, 5}));
    EXPECT_EQ(finiteCheck.getTensorLabel(), "after_projection");
    EXPECT_TRUE(finiteCheck.getCheckForward());
    EXPECT_FALSE(finiteCheck.getCheckBackward());
    EXPECT_FALSE(finiteCheck.getFailOnNonFinite());
    EXPECT_EQ(finiteCheck.getMaxReportedIndices(), 4U);
}

TEST(UtilityApiLayers, FiniteCheckRejectsInvalidConfiguration) {
    Network network("finiteCheckInvalid");
    Tensor featureInput(DataType::FP32, {4});

    EXPECT_THROW(FiniteCheck::Builder()
                     .network(network)
                     .featureInput(featureInput)
                     .checkForward(false)
                     .checkBackward(false)
                     .build(),
                 invalid_argument);

    EXPECT_THROW(FiniteCheck::Builder()
                     .network(network)
                     .featureInput(featureInput)
                     .maxReportedIndices(ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES + 1)
                     .build(),
                 invalid_argument);
}

TEST(UtilityApiLayers, FiniteCheckArchitecturePersistsDiagnosticPolicy) {
    Network network("finiteCheckArchitecture");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({4})
                             .dataType(DataType::FP32)
                             .build();
    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .tensorLabel("encoder_output")
                                  .checkForward(true)
                                  .checkBackward(true)
                                  .failOnNonFinite(true)
                                  .maxReportedIndices(7)
                                  .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(finiteCheck.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();

    const json architecture = finiteCheck.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "finite_check");
    EXPECT_EQ(architecture.at("tensor_label").get<string>(), "encoder_output");
    EXPECT_TRUE(architecture.at("check_forward").get<bool>());
    EXPECT_TRUE(architecture.at("check_backward").get<bool>());
    EXPECT_TRUE(architecture.at("fail_on_non_finite").get<bool>());
    EXPECT_EQ(architecture.at("max_reported_indices").get<uint32_t>(), 7U);
}
