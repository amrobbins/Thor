#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

using namespace Thor;

namespace {

class InspectableNetwork : public Network {
   public:
    using Network::Network;

    uint64_t sumAllFirstInstanceBytes(uint32_t batchSize, ThorImplementation::TensorPlacement placement) const {
        uint64_t bytes = 0;
        for (const std::shared_ptr<Layer>& layer : network) {
            bytes += layer->getFirstInstanceMemRequirementInBytes(batchSize, placement);
        }
        return bytes;
    }

    StatusCode buildInferenceDag() { return connect(/*inferenceOnly=*/true); }

    uint64_t inferenceFirstInstanceBytes(uint32_t batchSize, ThorImplementation::TensorPlacement placement) {
        return computeFirstInstanceMemRequirements(batchSize, placement);
    }
};

class DeliberatelyOverestimatingStub : public Stub {
   public:
    DeliberatelyOverestimatingStub(Network& network, Tensor input) {
        featureInput = std::move(input);
        initialized = true;
        addToNetwork(&network);
    }

    DeliberatelyOverestimatingStub(const DeliberatelyOverestimatingStub&) = default;

    std::shared_ptr<Layer> clone() const override {
        return std::make_shared<DeliberatelyOverestimatingStub>(*this);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(
        uint32_t batchSize,
        ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        // This deliberately exceeds any practical GPU capacity. The API-layer
        // estimate is advisory; actual placement must be decided by physical
        // allocations and the serialized startup transaction's safety reserve.
        return uint64_t{1} << 62;
    }
};

}  // namespace

TEST(NetworkMemoryRequirements, ConservativeApiEstimateDoesNotRejectPhysicalPlacementThatFits) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "CUDA GPU required for placement regression.";
    }

    Network network("advisory_memory_estimate");
    NetworkInput input = NetworkInput::Builder()
                             .network(network)
                             .name("input")
                             .dimensions({1})
                             .dataType(DataType::FP32)
                             .build();
    DeliberatelyOverestimatingStub sink(network, input.getFeatureOutput().value());
    (void)sink;

    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(/*batchSize=*/1,
                                           initDoneEvents,
                                           /*inferenceOnly=*/true));
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) {
        event.synchronize();
    }
    placed->synchronize();
}

TEST(NetworkMemoryRequirements, LoadedInferenceExcludesPrunedTrainingOnlyBranches) {
    Network trainingNetwork("inference_memory_pruning_source");
    NetworkInput predictions = NetworkInput::Builder()
                                   .network(trainingNetwork)
                                   .name("predictions")
                                   .dimensions({1024})
                                   .dataType(DataType::FP32)
                                   .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(trainingNetwork)
                              .name("labels")
                              .dimensions({1024})
                              .dataType(DataType::FP32)
                              .build();

    NetworkOutput::Builder()
        .network(trainingNetwork)
        .name("prediction_output")
        .inputTensor(predictions.getFeatureOutput().value())
        .build();

    MSE loss = MSE::Builder()
                   .network(trainingNetwork)
                   .predictions(predictions.getFeatureOutput().value())
                   .labels(labels.getFeatureOutput().value())
                   .reportsRawLoss()
                   .build();
    NetworkOutput::Builder().network(trainingNetwork).name("loss_output").inputTensor(loss.getLoss()).build();

    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path archiveDir =
        std::filesystem::temp_directory_path() / (std::string("thor_inference_memory_pruning_") + std::to_string(now));
    std::filesystem::create_directories(archiveDir);

    trainingNetwork.save(archiveDir.string(), true);

    InspectableNetwork loaded("inference_memory_pruning_source");
    loaded.load(archiveDir.string());

    constexpr uint32_t batchSize = 8;
    ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::GPU, 0);

    const uint64_t unprunedBytes = loaded.sumAllFirstInstanceBytes(batchSize, placement);
    ASSERT_EQ(loaded.buildInferenceDag(), Network::StatusCode::SUCCESS);
    const uint64_t inferenceBytes = loaded.inferenceFirstInstanceBytes(batchSize, placement);

    EXPECT_GT(unprunedBytes, inferenceBytes)
        << "Inference memory admission must exclude the loaded artifact's pruned loss/label-only branch.";

    std::filesystem::remove_all(archiveDir);
}
