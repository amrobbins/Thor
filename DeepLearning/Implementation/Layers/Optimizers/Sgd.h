#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

namespace ThorImplementation {

class Sgd : public Optimizer {
   public:
    Sgd(std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer,
        float initialLearningRate,
        float decay,
        float momentum,
        bool useNesterovMomentum,
        Optional<Tensor> errorInput,
        Optional<Tensor> errorOutput);

    virtual void computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues);
    virtual void updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batch_size);

    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual std::unordered_map<std::string, float> getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual void setInitialLearningRate(float initaialLearningRate);
    virtual void setDecay(float decay);
    virtual void setMomentum(float momentum);
    virtual void setUseNesterovMomentum(bool useNesterovMomentum);

    virtual float getInitialLearningRate();
    virtual float getDecay();
    virtual float getMomentum();
    virtual bool getUseNesterovMomentum();

   protected:
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterovMomentum;

    float currentLearningRate;

    uint32_t epoch;

    uint32_t gpuNum;

    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayerShared;
    TrainableWeightsBiasesLayer *trainableLayer;

    uint32_t numInputFeatures;
    uint32_t numOutputputFeatures;

    Tensor previousWeightsUpdate;
    Tensor previousBiasesUpdate;
};

}  // namespace ThorImplementation