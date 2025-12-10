#pragma once

#include <cuda_fp16.h>
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/Optimizers/Adam.h"

#include <unordered_map>

namespace ThorImplementation {

class Adam : public Optimizer {
   public:
    Adam(std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer, float alpha, float beta1, float beta2, float epsilon);

    virtual void compile();

    virtual void computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues);

    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual std::unordered_map<std::string, float> getAllHyperParameters();

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

    // For testing or research purposes:
    virtual Tensor getM();
    virtual Tensor getV();
    virtual Optional<Tensor> getMBias();
    virtual Optional<Tensor> getVBias();

    virtual void dumpMToFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());
    virtual void dumpVToFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());
    virtual void dumpMBiasToFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());
    virtual void dumpVBiasToFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());

    virtual void loadMFromFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());
    virtual void loadVFromFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());
    virtual void loadMBiasFromFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());
    virtual void loadVBiasFromFile(std::string filename, Optional<Stream> stream = Optional<Stream>::empty());

   protected:
    void loadParamsFromFiles();

    float t;
    float alpha;
    float beta1;
    float beta2;
    float epsilon;

    Tensor m;
    Tensor v;
    Optional<Tensor> mBias;
    Optional<Tensor> vBias;

    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayerShared;
    TrainableWeightsBiasesLayer *trainableLayer;

    uint32_t gpuNum;

    Optional<std::string> mFile;
    Optional<std::string> vFile;
    Optional<std::string> mBiasFile;
    Optional<std::string> vBiasFile;

    bool compiled = false;
};

}  // namespace ThorImplementation
