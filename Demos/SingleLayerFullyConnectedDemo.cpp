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

int main() {
    Thor::Network singleLayerFullyConnected = buildSingleLayerFullyConnected();

    set<string> shardPaths;

    // home/andrew/mnist/raw
    // test_images.bin  test_labels.bin  train_images.bin  train_labels.bin
    // These are raw 1 byte pixels of 28x28 images and raw 1 byte labels
    // Need to get these into a shard

    assert(boost::filesystem::exists("/media/andrew/PCIE_SSD/Mnist_1_of_1.shard"));
    shardPaths.insert("/media/andrew/PCIE_SSD/Mnist_1_of_1.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::FP32, {28 * 28});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, 48);
    batchLoader->setDatasetName("MNIST");

    std::shared_ptr<Sgd> sgd = Sgd::Builder().initialLearningRate(0.1).decay(0.6).momentum(0.0).build();

    shared_ptr<Thor::LocalExecutor> executor = LocalExecutor::Builder()
                                                   .network(singleLayerFullyConnected)
                                                   .loader(batchLoader)
                                                   .optimizer(sgd)
                                                   .visualizer(&ConsoleVisualizer::instance())
                                                   .build();

    set<string> tensorsToReturn;
    tensorsToReturn.insert("predictions");
    tensorsToReturn.insert("loss");
    tensorsToReturn.insert("accuracy");

    executor->trainEpochs(200, tensorsToReturn);

    return 0;
}
