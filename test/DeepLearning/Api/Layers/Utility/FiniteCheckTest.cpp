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
                                  .enabled(false)
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
    EXPECT_FALSE(finiteCheck.getEnabled());
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
                                  .enabled(false)
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
    EXPECT_FALSE(architecture.at("enabled").get<bool>());
    EXPECT_TRUE(architecture.at("check_forward").get<bool>());
    EXPECT_TRUE(architecture.at("check_backward").get<bool>());
    EXPECT_TRUE(architecture.at("fail_on_non_finite").get<bool>());
    EXPECT_EQ(architecture.at("max_reported_indices").get<uint32_t>(), 7U);
}

TEST(UtilityApiLayers, RaggedFiniteCheckPreservesExactPartitionAndSerializes) {
    Network network("raggedFiniteCheckBuilds");
    Tensor values(DataType::FP32, {8, 2});
    Tensor offsets(DataType::UINT64, {4});
    RaggedTensor input(values, offsets, 5);

    FiniteCheck finiteCheck = FiniteCheck::Builder()
                                  .network(network)
                                  .featureInput(input)
                                  .tensorLabel("ragged_history")
                                  .enabled(false)
                                  .build();

    ASSERT_TRUE(finiteCheck.getUseRagged());
    ASSERT_TRUE(finiteCheck.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(finiteCheck.getRaggedFeatureOutput().has_value());
    const RaggedTensor output = finiteCheck.getRaggedFeatureOutput().value();
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getBatchSize(), input.getBatchSize());
    EXPECT_EQ(output.getMaxTotalValues(), input.getMaxTotalValues());
    ASSERT_TRUE(output.hasMaxValuesPerRow());
    EXPECT_EQ(output.getMaxValuesPerRow(), 5u);
    EXPECT_EQ(output.getValues().getDimensions(), values.getDimensions());
    EXPECT_NE(output.getValues(), values);

    const json architecture = finiteCheck.architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(), offsets.getId());
    EXPECT_EQ(architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(), offsets.getId());

    auto cloned = dynamic_pointer_cast<FiniteCheck>(finiteCheck.clone());
    ASSERT_NE(cloned, nullptr);
    ASSERT_TRUE(cloned->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(cloned->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(cloned->getRaggedFeatureInput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(cloned->getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
}
