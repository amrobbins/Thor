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
using namespace std;

int main() {
    set<string> shardPaths;

    Network singleLayerConvolution2d = buildSingleLayerConvolution2d();

    assert(boost::filesystem::exists("/PCIE_SSD/Mnist_1_of_1.shard"));
    shardPaths.insert("/PCIE_SSD/Mnist_1_of_1.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::FP32, {1, 28, 28});
    ThorImplementation::TensorDescriptor labelDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {10});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, labelDescriptor, 48);
    batchLoader->setDatasetName("MNIST");

    std::shared_ptr<Optimizer> sgd =
        Sgd::Builder().network(singleLayerConvolution2d).initialLearningRate(0.1).decay(0.4).momentum(0.0).build();

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
