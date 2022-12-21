#include "Layer.h"
#include "NeuralNetwork/DropOut.h"

using namespace ThorImplementation;
using namespace std;

atomic<uint64_t> Layer::nextId(2);

mutex DropOut::mtx;
uint64_t DropOut::seed = 0ul;

cudnnTensorDescriptor_t Layer::createCudnnTensorDescriptor(vector<unsigned long> featureInputDimensions,
                                                           TensorDescriptor::DataType dataType) {
    cudnnTensorDescriptor_t descriptor;

    cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    // Tensors must have at least 4 dimensions and not more than CUDNN_DIM_MAX, per cudnn.
    // Unused dimensions will be set to size 1.
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetTensorNdDescriptor
    assert(featureInputDimensions.size() <= CUDNN_DIM_MAX);
    vector<int> dimensionsMin4;
    vector<int> noGapsStride;
    for (unsigned int i = 0; i < featureInputDimensions.size(); ++i) {
        dimensionsMin4.push_back(featureInputDimensions[i]);
        // no overflow:
        assert(dimensionsMin4.back() == (long)featureInputDimensions[i]);
        noGapsStride.push_back(1);
    }

    while (dimensionsMin4.size() < 4) {
        dimensionsMin4.push_back(1);
        noGapsStride.push_back(1);
    }

    for (int i = (int)dimensionsMin4.size() - 2; i >= 0; --i) {
        noGapsStride[i] = noGapsStride[i + 1] * dimensionsMin4[i + 1];
    }

    cudnnStatus = cudnnSetTensorNdDescriptor(
        descriptor, CudnnHelper::getCudnnDataType(dataType), dimensionsMin4.size(), dimensionsMin4.data(), noGapsStride.data());
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

    return descriptor;
}
