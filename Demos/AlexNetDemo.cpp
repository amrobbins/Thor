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
    Thor::Network alexNet = buildAlexNet();

    set<string> shardPaths;

    assert(boost::filesystem::exists("/media/andrew/PCIE_SSD/ImageNet2012_1_of_2.shard"));
    assert(boost::filesystem::exists("/media/andrew/PCIE_SSD/ImageNet2012_2_of_2.shard"));
    shardPaths.insert("/media/andrew/PCIE_SSD/ImageNet2012_1_of_2.shard");
    shardPaths.insert("/media/andrew/PCIE_SSD/ImageNet2012_2_of_2.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, 512);
    batchLoader->setDatasetName("ImageNet 2012");

    std::shared_ptr<Sgd> sgd = Sgd::Builder().initialLearningRate(0.01).decay(0.9).momentum(0).build();

    shared_ptr<Thor::LocalExecutor> executor =
        LocalExecutor::Builder().network(alexNet).loader(batchLoader).optimizer(sgd).visualizer(&ConsoleVisualizer::instance()).build();

    set<string> tensorsToReturn;
    tensorsToReturn.insert("predictions");
    tensorsToReturn.insert("loss");

    executor->trainEpochs(2, tensorsToReturn);

    // executor->createSnapshot("/media/andrew/PCIE_SSD/alexnetSnapshot_epoch5");

    return 0;
}
