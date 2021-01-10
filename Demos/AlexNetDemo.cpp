// g++ -Wall -Werror -fopenmp -ggdb -o AlexNetDemo -O3 -std=c++11 -pthread AlexNetDemo.cpp -I /usr/local/cuda/include -I /usr/include -I
// /usr/local/cuda/include -I /usr/include -I/home/andrew/Thor -L/home/andrew/Thor -L /usr/local/cuda/lib64 -l cublas -l cublasLt -l cusolver
// -l cudart -L /usr/lib/x86_64-linux-gnu -l cudnn /usr/local/lib/libboost_filesystem.a -I./ -L./ -lThor -I ./ -I /usr/local/cuda/include -I
// /usr/include -I /usr/local/boost -ldl -I /usr/local/include/GraphicsMagick/ -I build/test/googletest/include -pthread -I
// /usr/local/cuda/include -I /usr/include -L /usr/local/cuda/lib64 -l cublas -l cublasLt -l cusolver -l cudart -L /usr/lib/x86_64-linux-gnu
// -l cudnn /usr/local/lib/libboost_filesystem.a `GraphicsMagick++-config --cppflags --cxxflags --ldflags --libs`
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

int main() {
    /*
        vector<shared_ptr<Shard>> shards;
        shards.push_back(make_shared<Shard>());
        shards.push_back(make_shared<Shard>());
        shards[0]->openShard("/media/andrew/PCIE_SSD/ImageNet2012_1_of_2.shard");
        shards[1]->openShard("/media/andrew/PCIE_SSD/ImageNet2012_2_of_2.shard");

        printf("%ld %ld %ld %ld\n", shards[0]->getNumExamples(ExampleType::TRAIN), shards[0]->getNumExamples(ExampleType::VALIDATE),
       shards[0]->getNumExamples(ExampleType::TEST), shards[0]->getExampleSizeInBytes()); printf("%ld %ld %ld %ld\n",
       shards[1]->getNumExamples(ExampleType::TRAIN), shards[1]->getNumExamples(ExampleType::VALIDATE),
       shards[1]->getNumExamples(ExampleType::TEST), shards[1]->getExampleSizeInBytes());

        file_string_vector_t *allClasses0 = shards[0]->getAllClasses();
        file_string_vector_t *allClasses1 = shards[1]->getAllClasses();
        assert(allClasses0->size() == allClasses1->size());
        printf("num classes %ld\n", allClasses0->size());

        //for(uint64_t i = 0; i < allClasses0->size(); ++i) {
        //    printf("%s\n", (*allClasses0)[i].c_str());
        //}

        uint64_t batchSize = 512;
        BatchAssembler batchAssemblerTrain(
            shards,
            ExampleType::TRAIN,
            ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224}),
            batchSize);
        BatchAssembler batchAssemblerValidate(
            shards,
            ExampleType::VALIDATE,
            ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224}),
            batchSize);
        BatchAssembler batchAssemblerTest(
            shards,
            ExampleType::TEST,
            ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224}),
            batchSize);
        */

    Thor::Network alexNet = buildAlexNet();

    set<string> shardPaths;
    shardPaths.insert("/media/andrew/PCIE_SSD/ImageNet2012_1_of_2.shard");
    shardPaths.insert("/media/andrew/PCIE_SSD/ImageNet2012_2_of_2.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {3, 224, 224});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, 512);

    Thor::LocalExecutor executor = Thor::LocalExecutor::Builder()
                                       .network(alexNet)
                                       .loader(batchLoader)
                                       //.hyperparameterController(hyperparameterController)
                                       //.visualizer(consoleVisualizer)
                                       .build();

    executor.trainEpochs(5);
    executor.createSnapshot("/media/andrew/PCIE_SSD/alexnetSnapshot_epoch5");

    return 0;
}
