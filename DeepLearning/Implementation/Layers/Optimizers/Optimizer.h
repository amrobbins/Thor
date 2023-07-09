#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnHelper.h"

namespace ThorImplementation {

class Optimizer {
   public:
    Optimizer() {
        currentEpoch = 0;
        currentBatch = 0;
        id = nextId.fetch_add(1);
    }

    // Note: It is the responsibility of the layer to ensure all dependencies are available at the start of gradient update stream.
    //       And that the data stream will be blocked until
    virtual void computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) { assert(false); }
    virtual void updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) { assert(false); }

    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        assert(false);
    }
    virtual std::unordered_map<std::string, float> getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        assert(false);
    }

    // Whenever an incompatibility with Keras is created, this method must be used to catch the incompatibility and explain it to the user.
    virtual bool isKerasCompatible(std::string &explanation) {
        explanation.clear();
        return true;
    }

    virtual Stream getGradientUpdateStream() { return gradientUpdateStream; }

    virtual Tensor getWeightsGradient() { return weightsGradient; }
    virtual Optional<Tensor> getBiasesGradient() { return biasesGradient; }

    uint64_t getId() const { return id; }
    bool operator==(const Optimizer &other) const { return id == other.id; }

    virtual ~Optimizer() {}

    /**
     * C = alpha * A + beta * C
     */
    void accumulateScale(Tensor A, Tensor C, const void *alpha, const void *beta, Stream stream) {
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

        cudnnTensorDescriptor_t cudnnTensorADescriptor = Layer::createCudnnTensorDescriptor(dimensionsA, descriptorA.getDataType());
        cudnnTensorDescriptor_t cudnnTensorCDescriptor = Layer::createCudnnTensorDescriptor(dimensionsC, descriptorC.getDataType());

        cudnnStatus_t cudnnStatus = cudnnAddTensor(
            stream.getCudnnHandle(), alpha, cudnnTensorADescriptor, A.getMemPtr(), beta, cudnnTensorCDescriptor, C.getMemPtr());

        if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
            printf("cudnnStatus %d : %s\n", cudnnStatus, cudnnGetErrorString(cudnnStatus));
            fflush(stdout);
        }
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

   protected:
    uint64_t currentEpoch;
    uint64_t currentBatch;

    Stream gradientUpdateStream;

    Tensor weightsGradient;
    Optional<Tensor> biasesGradient;

   private:
    uint64_t id;
    static std::atomic<int64_t> nextId;
};

}  // namespace ThorImplementation