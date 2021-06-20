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

    ConsoleVisualizer::instance().startUI();

    set<string> shardPaths;

    assert(boost::filesystem::exists("/media/andrew/PCIE_SSD/ImageNet2012_1_of_2.shard"));
    assert(boost::filesystem::exists("/media/andrew/PCIE_SSD/ImageNet2012_2_of_2.shard"));
    shardPaths.insert("/media/andrew/PCIE_SSD/ImageNet2012_1_of_2.shard");
    shardPaths.insert("/media/andrew/PCIE_SSD/ImageNet2012_2_of_2.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, 512);

    std::shared_ptr<Sgd> sgd = Sgd::Builder().initialLearningRate(0.01).decay(0.9).momentum(0).build();

    shared_ptr<Thor::LocalExecutor> executor =
        LocalExecutor::Builder().network(alexNet).loader(batchLoader).optimizer(sgd).visualizer(&ConsoleVisualizer::instance()).build();

    set<string> tensorsToReturn;
    tensorsToReturn.insert("examples");
    tensorsToReturn.insert("labels");

    thread et(&LocalExecutor::trainEpoch, executor, ExampleType::TRAIN, tensorsToReturn);
    // executor->trainEpoch(ExampleType::TRAIN, tensorsToReturn);

    uint64_t i = 0;
    while (true) {
        executor->waitForBatchData();
        if (executor->isBatchDataReady())
            executor->popBatchData();
        else
            break;
        ++i;
        printf("%ld\n\r", i);
        fflush(stdout);
    }

    // executor->createSnapshot("/media/andrew/PCIE_SSD/alexnetSnapshot_epoch5");

    // cudaError_t cudaStatus = cudaDeviceSynchronize();
    // assert(cudaStatus == cudaSuccess);

    et.join();

    return 0;
}
