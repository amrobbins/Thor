#pragma once

#include <cuda_fp16.h>
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/Optimizers/Adam.h"

#include <unordered_map>

namespace ThorImplementation {

class Adam : public Optimizer {
   public:
    Adam(uint64_t id, std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer, float alpha, float beta1, float beta2, float epsilon);

    virtual void compile();

    using Optimizer::computeWeightsUpdate;
    virtual void computeWeightsUpdate(Tensor weightsGradient, Stream weightsGradientReadyStream, bool accumulateValues);
    virtual void stepFromPrecomputedGradient(bool accumulateValues);

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

    void testSetDataType(TensorDescriptor::DataType dataType) { weightsUpdateDataType = dataType; }

    static constexpr float MIN_FP16_EPSILON = 1.0e-4f;

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

    uint32_t gpuNum;

    Optional<std::string> mFile;
    Optional<std::string> vFile;
    Optional<std::string> mBiasFile;
    Optional<std::string> vBiasFile;
};

}  // namespace ThorImplementation
