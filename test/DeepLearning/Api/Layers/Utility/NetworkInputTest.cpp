#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;

TEST(NetworkInput, PassThroughSourceUsesApiTensorAsFeatureOutput) {
    Network network("network_input_api_pass_through_source");
    NetworkInput source = NetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .dimensions({3, 4})
                              .dataType(DataType::FP32)
                              .build();

    Tensor sourceTensor = source.getFeatureOutput().value();
    NetworkInput alias = NetworkInput::Builder()
                             .network(network)
                             .name("alias")
                             .passThroughSource(sourceTensor)
                             .build();

    ASSERT_TRUE(alias.hasPassThroughSource());
    EXPECT_EQ(alias.getPassThroughSource(), sourceTensor);
    EXPECT_EQ(alias.getFeatureOutput().value(), sourceTensor);
    EXPECT_EQ(alias.getFeatureInput().value(), sourceTensor);
    EXPECT_EQ(alias.getDimensions(), sourceTensor.getDimensions());
    EXPECT_EQ(alias.getDataType(), sourceTensor.getDataType());
}

TEST(NetworkInput, PassThroughSourceMayValidateExplicitShapeAndType) {
    Network network("network_input_api_pass_through_source_validation");
    NetworkInput source = NetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .dimensions({5})
                              .dataType(DataType::FP32)
                              .build();

    EXPECT_NO_THROW((NetworkInput::Builder()
                         .network(network)
                         .name("alias_ok")
                         .dimensions({5})
                         .dataType(DataType::FP32)
                         .passThroughSource(source.getFeatureOutput().value())
                         .build()));

    EXPECT_THROW((NetworkInput::Builder()
                      .network(network)
                      .name("alias_bad_shape")
                      .dimensions({6})
                      .dataType(DataType::FP32)
                      .passThroughSource(source.getFeatureOutput().value())
                      .build()),
                 std::logic_error);

    EXPECT_THROW((NetworkInput::Builder()
                      .network(network)
                      .name("alias_bad_dtype")
                      .dimensions({5})
                      .dataType(DataType::FP16)
                      .passThroughSource(source.getFeatureOutput().value())
                      .build()),
                 std::logic_error);
}

TEST(NetworkInput, PassThroughSourceIsNotAnExternalNetworkInput) {
    Network network("network_input_api_pass_through_source_external_names");
    NetworkInput source = NetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .dimensions({2})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput alias = NetworkInput::Builder()
                             .network(network)
                             .name("alias")
                             .passThroughSource(source.getFeatureOutput().value())
                             .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(alias.getFeatureOutput().value())
        .build();

    std::vector<std::string> inputNames = network.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "source");
}

TEST(NetworkInput, DuplicatePassThroughNamesAreIgnoredForExternalInputValidation) {
    Network network("network_input_api_pass_through_duplicate_names");
    NetworkInput source = NetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .dimensions({2})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput alias0 = NetworkInput::Builder()
                              .network(network)
                              .name("member_input")
                              .passThroughSource(source.getFeatureOutput().value())
                              .build();
    NetworkInput alias1 = NetworkInput::Builder()
                              .network(network)
                              .name("member_input")
                              .passThroughSource(source.getFeatureOutput().value())
                              .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output_0")
        .inputTensor(alias0.getFeatureOutput().value())
        .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output_1")
        .inputTensor(alias1.getFeatureOutput().value())
        .build();

    std::vector<std::string> inputNames = network.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "source");
}

TEST(NetworkInput, PassThroughSourceMustBelongToSameNetwork) {
    Network sourceNetwork("network_input_api_pass_through_source_network");
    Network destinationNetwork("network_input_api_pass_through_destination_network");
    NetworkInput source = NetworkInput::Builder()
                              .network(sourceNetwork)
                              .name("source")
                              .dimensions({2})
                              .dataType(DataType::FP32)
                              .build();

    EXPECT_THROW((NetworkInput::Builder()
                      .network(destinationNetwork)
                      .name("alias_from_other_network")
                      .passThroughSource(source.getFeatureOutput().value())
                      .build()),
                 std::runtime_error);

    Tensor unownedTensor(DataType::FP32, {2});
    EXPECT_THROW((NetworkInput::Builder()
                      .network(destinationNetwork)
                      .name("alias_from_unowned_tensor")
                      .passThroughSource(unownedTensor)
                      .build()),
                 std::runtime_error);
}

TEST(NetworkInput, PassThroughSourceIsExcludedFromPlacedExternalInputSignature) {
    Network network("network_input_api_pass_through_source_placed_signature");
    NetworkInput source = NetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .dimensions({2})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput alias = NetworkInput::Builder()
                             .network(network)
                             .name("member_0/source")
                             .passThroughSource(source.getFeatureOutput().value())
                             .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(alias.getFeatureOutput().value())
        .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = network.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    std::vector<std::string> inputNames = placed->getNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "source");
}

TEST(NetworkInput, PassThroughSourceSerializationRoundTripPreservesAliasBehavior) {
    Network network("network_input_api_pass_through_source_round_trip");
    NetworkInput source = NetworkInput::Builder()
                              .network(network)
                              .name("source")
                              .dimensions({2})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput alias = NetworkInput::Builder()
                             .network(network)
                             .name("member_0/source")
                             .passThroughSource(source.getFeatureOutput().value())
                             .build();
    NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(alias.getFeatureOutput().value())
        .build();

    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path archiveDir =
        std::filesystem::temp_directory_path() / (std::string("thor_network_input_alias_round_trip_") + std::to_string(now));
    std::filesystem::create_directories(archiveDir);

    network.save(archiveDir.string(), true);

    Network loaded("network_input_api_pass_through_source_round_trip");
    loaded.load(archiveDir.string());

    std::vector<std::string> inputNames = loaded.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "source");

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = loaded.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }

    std::vector<std::string> placedInputNames = placed->getNetworkInputNames();
    ASSERT_EQ(placedInputNames.size(), 1u);
    EXPECT_EQ(placedInputNames[0], "source");

    std::filesystem::remove_all(archiveDir);
}

TEST(NetworkInput, ApiPassThroughHasNoPhysicalBridgeMethodOrField) {
#ifndef SOURCE_DIR
    GTEST_SKIP() << "SOURCE_DIR is not defined for source grep regression test.";
#else
    const std::filesystem::path sourceRoot = SOURCE_DIR;
    const std::vector<std::filesystem::path> files = {
        sourceRoot / "DeepLearning/Api/Layers/Utility/NetworkInput.h",
        sourceRoot / "DeepLearning/Api/Layers/Utility/NetworkInput.cpp",
    };

    std::string contents;
    for (const std::filesystem::path& file : files) {
        std::ifstream in(file);
        ASSERT_TRUE(in.is_open()) << "Unable to open " << file;
        std::ostringstream buffer;
        buffer << in.rdbuf();
        contents += buffer.str();
        contents.push_back('\n');
    }

    EXPECT_EQ(contents.find("passThroughPhysicalSource"), std::string::npos);
    EXPECT_EQ(contents.find("physicalPassThrough"), std::string::npos);
    EXPECT_EQ(contents.find("std::optional<ThorImplementation::Tensor"), std::string::npos);
    EXPECT_EQ(contents.find("ThorImplementation::Tensor passThrough"), std::string::npos);
#endif
}
