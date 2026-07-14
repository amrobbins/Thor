#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Data/BatchFieldSource.h"
#include "DeepLearning/Api/Data/DeviceBatchReference.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

class CopyTensorDeviceBatchMaterializer : public Api::DeviceBatchMaterializer {
   public:
    explicit CopyTensorDeviceBatchMaterializer(Impl::Tensor source) : source(std::move(source)) {}

    Impl::TensorDescriptor getOutputDescriptor() const override { return source.getDescriptor(); }
    Impl::TensorPlacement getOutputPlacement() const override { return source.getPlacement(); }

    void enqueueMaterialization(Impl::Tensor& destination, Stream& destinationStream) const override {
        destination.copyFromAsync(source, destinationStream);
    }

   private:
    Impl::Tensor source;
};

void synchronizeEvents(std::vector<Event>& events) {
    for (Event& event : events) {
        event.synchronize();
    }
}

std::vector<float> readTensor(Impl::Tensor tensor) {
    Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor cpuTensor = tensor.getPlacement() == cpuPlacement ? tensor : tensor.clone(cpuPlacement);
    if (tensor.getPlacement() != cpuPlacement) {
        Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
        cpuTensor.copyFromAsync(tensor, stream);
        stream.synchronize();
    }
    const float* values = cpuTensor.getMemPtr<float>();
    return std::vector<float>(values, values + cpuTensor.getTotalNumElements());
}

}  // namespace

TEST(DeviceBatchReference, PlacedNetworkDispatchesReferenceBatchThroughNamedInput) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "DeviceBatchReference placed-network test requires a GPU";
    }

    constexpr uint32_t batchSize = 2;
    Api::Network network("device_batch_reference_dispatch");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("features")
                                  .dimensions({3})
                                  .dataType(Impl::DataType::FP32)
                                  .build();
    Api::NetworkOutput::Builder()
        .network(network)
        .name("prediction")
        .inputTensor(input.getFeatureOutput().value())
        .dataType(Impl::DataType::FP32)
        .build();

    std::vector<Event> initializationDone;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(
        batchSize,
        initializationDone,
        /*inferenceOnly=*/true,
        std::vector<int32_t>{0},
        /*forcedNumStampsPerGpu=*/1);
    ASSERT_NE(placed, nullptr);
    synchronizeEvents(initializationDone);

    const Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);
    placed->configureBatchInputSources({
        {"features", Api::BatchFieldSourceDescription::deviceReference(gpuPlacement)},
    });
    placed->preallocateInputSlots(2);
    placed->preallocateOutputSlots(2);
    placed->synchronize();

    std::shared_ptr<Impl::NetworkInput> physicalInput =
        placed->getStampedNetwork(0).getNamedInput("features");
    ASSERT_NE(physicalInput, nullptr);
    ASSERT_TRUE(physicalInput->isDeviceReferenceLoad());
    ASSERT_EQ(physicalInput->getNumInputSlots(), 2u);
    ASSERT_TRUE(physicalInput->getFeatureOutput().has_value());

    Impl::Tensor source(gpuPlacement, physicalInput->getFeatureOutput().value().getDescriptor());
    Stream setupStream(0);
    source.fill(7.0f, setupStream);
    setupStream.synchronize();

    auto materializer = std::make_shared<CopyTensorDeviceBatchMaterializer>(source);
    Batch batch;
    batch.insert("features", Api::DeviceBatchReference(materializer, batchSize));

    std::map<std::string, Impl::Tensor> outputs;
    std::map<std::string, Event> outputReadyEvents;
    Event done = placed->submitBatch(
        0,
        batch,
        outputs,
        outputReadyEvents,
        /*isInferenceOnly=*/true,
        /*reusableProcessingFinishedEvent=*/nullptr,
        /*waitForOutputsOnProcessingStream=*/true,
        /*submitTiming=*/nullptr,
        /*outputSlotIndex=*/1);
    done.synchronize();
    outputReadyEvents.at("prediction").synchronize();

    const std::vector<float> values = readTensor(outputs.at("prediction"));
    ASSERT_EQ(values.size(), batchSize * 3u);
    for (float value : values) {
        EXPECT_EQ(value, 7.0f);
    }
}
