#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace Api = Thor;

namespace {

filesystem::path makeBoundParameterArchiveDirectory(const string& suffix) {
    const auto unique = chrono::high_resolution_clock::now().time_since_epoch().count();
    filesystem::path path = filesystem::temp_directory_path() /
                            ("thor_bound_parameter_" + suffix + "_" + to_string(unique));
    filesystem::remove_all(path);
    filesystem::create_directories(path);
    return path;
}

shared_ptr<Api::Optimizer> makeBoundParameterOptimizer() {
    return Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
}

Api::FullyConnected addFullyConnected(Api::Network& network,
                                      const Api::Tensor& input,
                                      const shared_ptr<Api::Optimizer>& optimizer) {
    return Api::FullyConnected::Builder()
        .network(network)
        .featureInput(input)
        .numOutputFeatures(2)
        .hasBias(true)
        .weightsOptimizer(optimizer)
        .biasesOptimizer(optimizer)
        .noActivation()
        .build();
}

shared_ptr<Api::FullyConnected> findFullyConnected(Api::Network& network) {
    shared_ptr<Api::FullyConnected> result;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<Api::FullyConnected> candidate = dynamic_pointer_cast<Api::FullyConnected>(network.getLayer(i));
        if (candidate == nullptr)
            continue;
        if (result != nullptr)
            throw runtime_error("Expected exactly one FullyConnected layer in BoundParameter test network.");
        result = candidate;
    }
    return result;
}

TEST(BoundParameterApi, EnablingPlacedExpressionParameterWithoutCompiledBackwardEdgeIsRejected) {
    constexpr uint32_t batchSize = 2;
    Api::Network network("bound_parameter_reject_missing_backward_edge");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({3})
                                  .dataType(Api::DataType::FP32)
                                  .build();
    Api::FullyConnected fullyConnected =
        addFullyConnected(network, input.getFeatureOutput().value(), makeBoundParameterOptimizer());
    fullyConnected.getParameterSpecification("weights")->setTrainingInitiallyEnabled(false);
    fullyConnected.getParameterSpecification("biases")->setTrainingInitiallyEnabled(false);
    Api::NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(fullyConnected.getFeatureOutput().value())
        .dataType(Api::DataType::FP32)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents)
        event.synchronize();
    ASSERT_NE(placed, nullptr);

    Api::BoundParameter weights = placed->resolveParameterReference(fullyConnected.getParameterReference("weights"));
    Api::BoundParameter biases = placed->resolveParameterReference(fullyConnected.getParameterReference("biases"));
    ASSERT_FALSE(weights.isTrainingEnabled());
    ASSERT_FALSE(biases.isTrainingEnabled());

    EXPECT_THROW(weights.setTrainingEnabled(true), runtime_error);
    EXPECT_FALSE(weights.isTrainingEnabled());
    EXPECT_FALSE(biases.isTrainingEnabled());
}

TEST(BoundParameterApi, ReenablingExpressionParameterIsAllowedWhenBackwardEdgeWasCompiled) {
    constexpr uint32_t batchSize = 2;
    Api::Network network("bound_parameter_reenable_existing_backward_edge");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({3})
                                  .dataType(Api::DataType::FP32)
                                  .build();
    Api::FullyConnected fullyConnected =
        addFullyConnected(network, input.getFeatureOutput().value(), makeBoundParameterOptimizer());
    Api::GradientRivet gradientRivet =
        Api::GradientRivet::Builder().network(network).tensor(fullyConnected.getFeatureOutput().value()).build();
    Api::NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(gradientRivet.getFeatureOutput().value())
        .dataType(Api::DataType::FP32)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    for (Event& event : initDoneEvents)
        event.synchronize();
    ASSERT_NE(placed, nullptr);

    Api::BoundParameter weights = placed->resolveParameterReference(fullyConnected.getParameterReference("weights"));
    Api::BoundParameter biases = placed->resolveParameterReference(fullyConnected.getParameterReference("biases"));
    ASSERT_TRUE(weights.isTrainingEnabled());
    ASSERT_TRUE(biases.isTrainingEnabled());

    ASSERT_NO_THROW(weights.setTrainingEnabled(false));
    ASSERT_NO_THROW(biases.setTrainingEnabled(false));
    EXPECT_FALSE(weights.isTrainingEnabled());
    EXPECT_FALSE(biases.isTrainingEnabled());

    EXPECT_NO_THROW(weights.setTrainingEnabled(true));
    EXPECT_TRUE(weights.isTrainingEnabled());
    EXPECT_FALSE(biases.isTrainingEnabled());
}

TEST(BoundParameterApi, StatefulSavePersistsCurrentRequestedTrainingState) {
    constexpr uint32_t batchSize = 2;
    const string networkName = "bound_parameter_save_current_training_state";
    const filesystem::path archiveDirectory = makeBoundParameterArchiveDirectory("save_state");

    try {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("input")
                                      .dimensions({3})
                                      .dataType(Api::DataType::FP32)
                                      .build();
        Api::FullyConnected fullyConnected =
            addFullyConnected(network, input.getFeatureOutput().value(), makeBoundParameterOptimizer());
        Api::NetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(fullyConnected.getFeatureOutput().value())
            .dataType(Api::DataType::FP32)
            .build();

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
        for (Event& event : initDoneEvents)
            event.synchronize();
        ASSERT_NE(placed, nullptr);

        Api::BoundParameter weights = placed->resolveParameterReference(fullyConnected.getParameterReference("weights"));
        Api::BoundParameter biases = placed->resolveParameterReference(fullyConnected.getParameterReference("biases"));
        ASSERT_TRUE(weights.isTrainingEnabled());
        ASSERT_TRUE(biases.isTrainingEnabled());
        ASSERT_NO_THROW(weights.setTrainingEnabled(false));
        EXPECT_FALSE(weights.isTrainingEnabled());
        EXPECT_TRUE(biases.isTrainingEnabled());

        placed->save(archiveDirectory.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        Api::Network loaded(networkName);
        loaded.load(archiveDirectory.string());
        shared_ptr<Api::FullyConnected> loadedFullyConnected = findFullyConnected(loaded);
        ASSERT_NE(loadedFullyConnected, nullptr);
        shared_ptr<Api::ParameterSpecification> loadedWeights =
            loadedFullyConnected->getParameterSpecification("weights");
        shared_ptr<Api::ParameterSpecification> loadedBiases =
            loadedFullyConnected->getParameterSpecification("biases");
        ASSERT_NE(loadedWeights, nullptr);
        ASSERT_NE(loadedBiases, nullptr);
        EXPECT_FALSE(loadedWeights->isTrainingInitiallyEnabled());
        EXPECT_TRUE(loadedBiases->isTrainingInitiallyEnabled());
    } catch (...) {
        filesystem::remove_all(archiveDirectory);
        throw;
    }
    filesystem::remove_all(archiveDirectory);
}

}  // namespace
