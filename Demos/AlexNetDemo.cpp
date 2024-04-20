#include "Thor.h"

#include <boost/filesystem.hpp>

#include <assert.h>
#include <memory.h>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda_profiler_api.h>

using std::set;
using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

using namespace Thor;
using namespace std;

int main() {
    Thor::Network alexNet = buildAlexNet();
    // Thor::Network alexNet = buildDeepFullyConnected();

    // assert(cudaProfilerStop() == cudaSuccess);

    cudaDeviceReset();

    set<string> shardPaths;

    constexpr uint32_t NUM_CLASSES = 1000;
    const string datasetPath("/PCIE_SSD/ImageNet2012_10_classes_1_of_1.shard");
    assert(boost::filesystem::exists(datasetPath));
    shardPaths.insert(datasetPath);
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::FP16, {3, 224, 224});
    ThorImplementation::TensorDescriptor labelDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {NUM_CLASSES});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, labelDescriptor, 512);
    batchLoader->setDatasetName("ImageNet 2012");

    // std::shared_ptr<Sgd> optimizer = Sgd::Builder().network(alexNet).initialLearningRate(0.05).decay(0.2).momentum(0.0).build();
    // std::shared_ptr<Adam> optimizer = Adam::Builder().network(alexNet).build();
    std::shared_ptr<Sgd> optimizer = Sgd::Builder().network(alexNet).initialLearningRate(0.01).momentum(0.9).build();

    shared_ptr<Thor::LocalExecutor> executor = LocalExecutor::Builder()
                                                   .network(alexNet)
                                                   .loader(batchLoader)
                                                   .optimizer(optimizer)
                                                   .visualizer(&ConsoleVisualizer::instance())
                                                   .build();

    set<string> tensorsToReturn;
    tensorsToReturn.insert("predictions");
    tensorsToReturn.insert("loss");
    tensorsToReturn.insert("accuracy");

    executor->trainEpochs(50, tensorsToReturn);

    // executor->createSnapshot("/media/andrew/PCIE_SSD/alexnetSnapshot_epoch5");

    return 0;
}
