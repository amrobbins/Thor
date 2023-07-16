#pragma once

#include <cuda_fp16.h>
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/Optimizers/Adam.h"

#include <unordered_map>

namespace ThorImplementation {

class Adam : public Optimizer {
   public:
    Adam(std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer,
         float alpha,
         float beta1,
         float beta2,
         float epsilon,
         Optional<Tensor> errorInput,
         Optional<Tensor> errorOutput);

    virtual void initialize();

    virtual void computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues);
    virtual void updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batch_size);

    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual std::unordered_map<std::string, float> getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);

    virtual float getT();
    virtual float getAlpha();
    virtual float getBeta1();
    virtual float getBeta2();
    virtual float getEpsilon();

    virtual void setT(float t);
    virtual void setAlpha(float alpha);
    virtual void setBeta1(float beta1);
    virtual void setBeta2(float beta2);
    virtual void setEpsilon(float epsilon);

   protected:
    float t;
    float alpha;
    float beta1;
    float beta2;
    float epsilon;

    Tensor m;
    Tensor v;
    Optional<Tensor> mBias;
    Optional<Tensor> vBias;

    Tensor weightsUpdate;
    Optional<Tensor> biasesUpdate;

    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayerShared;
    TrainableWeightsBiasesLayer *trainableLayer;

    uint32_t gpuNum;
};

}  // namespace ThorImplementation