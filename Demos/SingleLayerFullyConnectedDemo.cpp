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
    Network singleLayerFullyConnected = buildSingleLayerFullyConnected();

    cudaDeviceReset();

    set<string> shardPaths;

    // home/andrew/mnist/raw
    // test_images.bin  test_labels.bin  train_images.bin  train_labels.bin
    // These are raw 1 byte pixels of 28x28 images and raw 1 byte labels
    // Need to get these into a shard

    assert(boost::filesystem::exists("/PCIE_SSD/Mnist_1_of_1.shard"));
    shardPaths.insert("/PCIE_SSD/Mnist_1_of_1.shard");
    ThorImplementation::TensorDescriptor exampleDescriptor(ThorImplementation::TensorDescriptor::DataType::FP32, {28 * 28});
    ThorImplementation::TensorDescriptor labelDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {10});

    std::shared_ptr<LocalBatchLoader> batchLoader = make_shared<LocalBatchLoader>(shardPaths, exampleDescriptor, labelDescriptor, 48);
    batchLoader->setDatasetName("MNIST");

    // std::shared_ptr<Sgd> optimizer =
    // Sgd::Builder().network(singleLayerFullyConnected).initialLearningRate(0.1).decay(0.2).momentum(0.0).build();
    std::shared_ptr<Adam> optimizer = Adam::Builder().network(singleLayerFullyConnected).build();

    shared_ptr<Thor::LocalExecutor> executor = LocalExecutor::Builder()
                                                   .network(singleLayerFullyConnected)
                                                   .loader(batchLoader)
                                                   .optimizer(optimizer)
                                                   .visualizer(&ConsoleVisualizer::instance())
                                                   .build();

    set<string> tensorsToReturn;
    tensorsToReturn.insert("predictions");
    tensorsToReturn.insert("loss");
    tensorsToReturn.insert("accuracy");

    executor->trainEpochs(200, tensorsToReturn);

    /*
        for(uint32_t i = 0; i < 200; ++i) {
            executor->trainEpochs(1, tensorsToReturn);

            ThorImplementation::NetworkInput *labelsInput = executor->stampedNetworks.back().inputNamed["labels"];
            ThorImplementation::Tensor labels_d = labelsInput->getFeatureOutput();
            ThorImplementation::Tensor labels_h = labels_d.clone(TensorPlacement::MemDevices::CPU);
            labels_h.copyFromAsync(labels_d, labelsInput->getStream());
            labelsInput->getStream().synchronize();

            uint8_t *labelArray = (uint8_t *)labels_h.getMemPtr();
            vector<uint32_t> labelVector;
            vector<uint8_t> bestLabelVector;
            uint32_t numClasses = 10;
            for (uint32_t b = 0; b < 6; ++b) {
                uint8_t bestLabel = labelArray[b * numClasses];
                labelVector.push_back(0);
                for (uint32_t c = 1; c < numClasses; ++c) {
                    uint8_t classLabel = labelArray[b * numClasses + c];
                    if (classLabel > bestLabel) {
                        labelVector.pop_back();
                        labelVector.push_back(c);
                        bestLabel = classLabel;
                    }
                }
                bestLabelVector.push_back(bestLabel);
            }
            printf("\rLabels:      ");
            for (uint32_t i = 0; i < labelVector.size(); ++i)
                printf("%d(%d) ", labelVector[i], (uint32_t)bestLabelVector[i]);
            printf("\n\n");
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        }
    */

    return 0;
}
