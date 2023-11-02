#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"

using namespace ThorImplementation;
using namespace std;

atomic<int64_t> Optimizer::nextId(2);

void Optimizer::updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) {
    updateWeightsWithScale(weights, biases, 1.0f / batchSize);
}

void Optimizer::updateWeightsWithScale(Tensor weights, Optional<Tensor> biases, float weightUpdateScalingFactor) {
    // The optimizer base takes care of reverse loss scaling so that the individual optimizers don't need to worry about it.
    const float ALPHA = weightUpdateScalingFactor / Loss::getLossScalingFactor();
    const float BETA = 1.0f;

    accumulateScale(weights, weightsUpdate, &ALPHA, &BETA, gradientUpdateStream);
    if (biases.isPresent()) {
        assert(biasesUpdate.isPresent());
        accumulateScale(biases, biasesUpdate, &ALPHA, &BETA, gradientUpdateStream);
    }
}

void Optimizer::accumulateScale(Tensor C, Tensor A, const void *alpha, const void *beta, Stream stream) {
    // Verify compatibility
    TensorDescriptor descriptorA = A.getDescriptor();
    TensorDescriptor descriptorC = C.getDescriptor();
    std::vector<uint64_t> dimensionsA = A.getDimensions();
    std::vector<uint64_t> dimensionsC = C.getDimensions();

    assert(dimensionsA.size() > 0);
    assert(dimensionsA.size() <= 5);
    assert(dimensionsC.size() > 0);
    assert(dimensionsC.size() <= 5);
    assert(dimensionsA.size() == dimensionsC.size());

    assert(descriptorA.getDataType() == descriptorC.getDataType());

    for (uint32_t i = 0; i < dimensionsA.size(); ++i) {
        assert(dimensionsA[i] > 0);
        assert(dimensionsC[i] > 0);
        assert(dimensionsA[i] == dimensionsC[i] || dimensionsA[i] == 1);
    }

    if (descriptorA != previousDescriptorA) {
        previousDescriptorA = descriptorA;
        cudnnTensorDescriptorA = Layer::createCudnnTensorDescriptor(dimensionsA, descriptorA.getDataType());
    }
    if (descriptorC != previousDescriptorC) {
        previousDescriptorC = descriptorC;
        cudnnTensorDescriptorC = Layer::createCudnnTensorDescriptor(dimensionsC, descriptorC.getDataType());
    }

    cudnnStatus_t cudnnStatus =
        cudnnAddTensor(stream.getCudnnHandle(), alpha, cudnnTensorDescriptorA, A.getMemPtr(), beta, cudnnTensorDescriptorC, C.getMemPtr());

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("cudnnStatus %d : %s\n", cudnnStatus, cudnnGetErrorString(cudnnStatus));
        fflush(stdout);
    }
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

Tensor Optimizer::getWeightsUpdate() { return weightsUpdate; }
Optional<Tensor> Optimizer::getBiasesUpdate() { return biasesUpdate; }