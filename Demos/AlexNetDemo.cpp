// g++ -Wall -Werror -fopenmp -ggdb -o AlexNetDemo -O3 -std=c++11 -pthread Demos/AlexNetDemo.cpp -I /usr/local/cuda/include -I /usr/include
// -I /usr/local/cuda/include -I /usr/include -I/home/andrew/Thor -L/home/andrew/Thor -L /usr/local/cuda/lib64 -l cublas -l cublasLt -l
// cusolver -l cudart -L /usr/lib/x86_64-linux-gnu -l cudnn /usr/local/lib/libboost_filesystem.a -I./ -L./ -lThor -I ./ -I
// /usr/local/cuda/include -I /usr/include -I /usr/local/boost -ldl -I /usr/local/include/GraphicsMagick/ -I build/test/googletest/include
// -pthread -I /usr/local/cuda/include -I /usr/include -L /usr/local/cuda/lib64 -l cublas -l cublasLt -l cusolver -l cudart -l ncurses -L
// /usr/lib/x86_64-linux-gnu -l cudnn /usr/local/lib/libboost_filesystem.a `GraphicsMagick++-config --cppflags --cxxflags --ldflags --libs`

#include "Thor.h"

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
