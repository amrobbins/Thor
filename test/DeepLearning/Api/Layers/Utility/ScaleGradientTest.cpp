#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, ScaleGradientBuildsAndSerializesScale) {
    Network network("scale_gradient_builds");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({4})
                             .dataType(DataType::FP32)
                             .build();

    ScaleGradient scaleGradient = ScaleGradient::Builder()
                                      .network(network)
                                      .featureInput(input.getFeatureOutput().value())
                                      .scale(-0.25f)
                                      .build();

    ASSERT_TRUE(scaleGradient.isInitialized());
    EXPECT_FLOAT_EQ(scaleGradient.getScale(), -0.25f);
    ASSERT_TRUE(scaleGradient.getFeatureOutput().has_value());
    EXPECT_EQ(scaleGradient.getFeatureOutput().value().getDimensions(), vector<uint64_t>({4}));
    EXPECT_EQ(scaleGradient.getFeatureOutput().value().getDataType(), DataType::FP32);
    EXPECT_NE(scaleGradient.getFeatureOutput().value(), input.getFeatureOutput().value());

    json architecture = scaleGradient.architectureJson();
    EXPECT_EQ(architecture.at("layer_type").get<string>(), "scale_gradient");
    EXPECT_FLOAT_EQ(architecture.at("scale").get<float>(), -0.25f);
}

TEST(UtilityApiLayers, ScaleGradientSerializeDeserializeAndStamp) {
    Network initialNetwork("scale_gradient_initial");
    NetworkInput input = NetworkInput::Builder()
                             .network(initialNetwork)
                             .name("input")
                             .dimensions({3})
                             .dataType(DataType::FP32)
                             .build();
    ScaleGradient scaleGradient = ScaleGradient::Builder()
                                      .network(initialNetwork)
                                      .featureInput(input.getFeatureOutput().value())
                                      .scale(0.1f)
                                      .build();
    NetworkOutput output = NetworkOutput::Builder()
                               .network(initialNetwork)
                               .name("output")
                               .inputTensor(scaleGradient.getFeatureOutput().value())
                               .dataType(DataType::FP32)
                               .build();

    Stream stream(0);
    thor_file::TarWriter archiveWriter("scaleGradientModel");
    json inputJ = input.serialize(archiveWriter, stream);
    json scaleJ = scaleGradient.serialize(archiveWriter, stream);
    json outputJ = output.serialize(archiveWriter, stream);

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor dummyData(cpuPlacement, ThorImplementation::TensorDescriptor(ThorImplementation::DataType::UINT8, {1}));
    archiveWriter.addArchiveFile("dummy", dummyData);
    archiveWriter.createArchive("/tmp/", true);

    Network loadedNetwork("scale_gradient_loaded");
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("scaleGradientModel", "/tmp/");
    Layer::deserialize(archiveReader, inputJ, &loadedNetwork);
    Layer::deserialize(archiveReader, scaleJ, &loadedNetwork);
    Layer::deserialize(archiveReader, outputJ, &loadedNetwork);

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed = loadedNetwork.place(2, initDoneEvents);
    ASSERT_NE(placed, nullptr);
    for (Event &event : initDoneEvents)
        stream.waitEvent(event);

    ThorImplementation::StampedNetwork stampedNetwork = placed->getStampedNetwork(0);
    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    auto stamped = dynamic_pointer_cast<ThorImplementation::ScaleGradient>(otherLayers[0]);
    ASSERT_NE(stamped, nullptr);
    EXPECT_FLOAT_EQ(stamped->getScale(), 0.1f);
}

TEST(UtilityApiLayers, ScaleGradientBuilderRejectsNonFiniteScale) {
    Network network("scale_gradient_bad_scale");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({4})
                             .dataType(DataType::FP32)
                             .build();

    EXPECT_THROW(ScaleGradient::Builder()
                     .network(network)
                     .featureInput(input.getFeatureOutput().value())
                     .scale(std::numeric_limits<float>::infinity()),
                 std::logic_error);
}
