#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
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

    virtual void compile() { compiled = true; }

    // Note: It is the responsibility of the layer to ensure all dependencies are available at the start of gradient update stream.
    //       And that the data stream will be blocked until
    virtual void computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) { assert(false); }
    virtual void updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize);
    virtual void updateWeightsWithScale(Tensor weights, Optional<Tensor> biases, float weightUpdateScalingFactor);

    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        assert(false);
    }
    virtual std::unordered_map<std::string, float> getAllHyperParameters() { assert(false); }

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
    void accumulateScale(Tensor C, Tensor A, const void *alpha, const void *beta, Stream stream);

    // For testing or research purposes:
    virtual Tensor getWeightsUpdate();
    virtual Optional<Tensor> getBiasesUpdate();

   protected:
    uint64_t currentEpoch;
    uint64_t currentBatch;

    Stream gradientUpdateStream;

    Tensor weightsGradient;
    Optional<Tensor> biasesGradient;

    Tensor weightsUpdate;
    Optional<Tensor> biasesUpdate;

    bool compiled = false;

   private:
    uint64_t id;
    static std::atomic<int64_t> nextId;

    cudnnTensorDescriptor_t cudnnTensorDescriptorA;
    cudnnTensorDescriptor_t cudnnTensorDescriptorC;
    TensorDescriptor previousDescriptorA;
    TensorDescriptor previousDescriptorC;
};

}  // namespace ThorImplementation
