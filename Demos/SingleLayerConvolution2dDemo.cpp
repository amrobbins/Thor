#include "Thor.h"

#include <boost/filesystem.hpp>

#include <assert.h>
#include <memory.h>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using std::set;
using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

using namespace Thor;

Network buildSingleLayerConvolution2d() {
    Network singleLayerConvolution2d;
    singleLayerConvolution2d.setNetworkName("singleLayerConvolution2d");

    vector<uint64_t> expectedDimensions;

    Glorot::Builder glorot = Glorot::Builder();

    NetworkInput imagesInput = NetworkInput::Builder()
                                   .network(singleLayerConvolution2d)
                                   .name("examples")
                                   .dimensions({1, 28, 28})
                                   .dataType(Tensor::DataType::FP32)
                                   .build();

    Tensor latestOutputTensor;
    latestOutputTensor = Convolution2d::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(imagesInput.getFeatureOutput())
                             .numOutputChannels(128)
                             .filterHeight(28)
                             .filterWidth(28)
                             .verticalStride(1)
                             .horizontalStride(1)
                             .noPadding()
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .activationBuilder(Relu::Builder())
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {128, 1, 1};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    latestOutputTensor = FullyConnected::Builder()
                             .network(singleLayerConvolution2d)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(10)
                             .hasBias(true)
                             .weightsInitializerBuilder(glorot)
                             .biasInitializerBuilder(glorot)
                             .noActivation()
                             .build()
                             .getFeatureOutput();
    expectedDimensions = {10};
    assert(latestOutputTensor.getDimensions() == expectedDimensions);

    Tensor labelsTensor = NetworkInput::Builder()
                              .network(singleLayerConvolution2d)
                              .name("labels")
                              .dimensions({10})
                              .dataType(Tensor::DataType::UINT8)
                              .build()
                              .getFeatureOutput();

    CategoricalCrossEntropyLoss lossLayer = CategoricalCrossEntropyLoss::Builder()
                                                .network(singleLayerConvolution2d)
                                                .featureInput(latestOutputTensor)
                                                .labels(labelsTensor)
                                                .lossType(ThorImplementation::Loss::ConnectionType::BATCH_LOSS)
                                                .build();

    labelsTensor = lossLayer.getLabels();

    NetworkOutput predictions = NetworkOutput::Builder()
                                    .network(singleLayerConvolution2d)
                                    .name("predictions")
                                    .inputTensor(lossLayer.getPredictions())
                                    .dataType(Tensor::DataType::FP32)
                                    .build();
    NetworkOutput loss = NetworkOutput::Builder()
                             .network(singleLayerConvolution2d)
                             .name("loss")
                             .inputTensor(lossLayer.getLoss())
                             .dataType(Tensor::DataType::FP32)
                             .build();

    CategoricalAccuracy accuracyLayer = CategoricalAccuracy::Builder()
                                            .network(singleLayerConvolution2d)
                                            .predictions(lossLayer.getPredictions())
                                            .labels(labelsTensor)
                                            .build();

    NetworkOutput accuracy = NetworkOutput::Builder()
                                 .network(singleLayerConvolution2d)
                                 .name("accuracy")
                                 .inputTensor(accuracyLayer.getMetric())
                                 .dataType(Tensor::DataType::FP32)
                                 .build();

    // Return the assembled network
    return singleLayerConvolution2d;
}


int main() {
    Network singleLayerConvolution2d = buildSingleLayerConvolution2d();

    cudaDeviceReset();

    set<string> shardPaths;

    // home/andrew/mnist/raw
    // test_images.bin  test_labels.bin  train_images.bin  train_labels.bin
    // These are raw 1 byte pixels of 28x28 images and raw 1 byte labels
    // These have been put into a shard

    assert(boost::filesystem::exists("/media/andrew/PCIE_SSD/Mnist_1_of_1.shard"));
    shardPaths.insert("/media/andrew/PCIE_SSD/Mnist_1_of_1.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::FP32, {1, 28, 28});
    ThorImplementation::TensorDescriptor labelDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {10});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, labelDescriptor, 48);
    batchLoader->setDatasetName("MNIST");

    std::shared_ptr<Sgd> sgd = Sgd::Builder().initialLearningRate(0.1).decay(0.6).momentum(0.0).build();

    shared_ptr<Thor::LocalExecutor> executor = LocalExecutor::Builder()
                                                   .network(singleLayerConvolution2d)
                                                   .loader(batchLoader)
                                                   .optimizer(sgd)
                                                   .visualizer(&ConsoleVisualizer::instance())
                                                   .build();

    set<string> tensorsToReturn;
    tensorsToReturn.insert("predictions");
    tensorsToReturn.insert("loss");
    tensorsToReturn.insert("accuracy");

    executor->trainEpochs(50, tensorsToReturn);

    return 0;
}
