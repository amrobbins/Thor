#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

namespace ThorImplementation {

class UniformRandomInitializer : public Initializer {
   public:
    UniformRandomInitializer(double maxValue, double minValue) : maxValue(maxValue), minValue(minValue) {
        assert(isfinite(maxValue));
        assert(isfinite(minValue));
        assert(maxValue >= minValue);
    }

    virtual void initialize(Layer *layer, Tensor tensorToInitialize) const { Initializer::initialize(layer, tensorToInitialize); }

    virtual void initialize(vector<Stream> streams, Tensor tensorToInitialize, bool synchronous) const {
        Tensor buffer = tensorToInitialize.clone(TensorPlacement::MemDevices::CPU);

        std::uniform_real_distribution<float> distribution(minValue, maxValue);
        std::hash<int> threadNumHash;

        uint64_t totalNumWeights = tensorToInitialize.getDescriptor().getTotalNumElements();
        half *bufferMem = (half *)buffer.getMemPtr();
        int numProcessors = omp_get_num_procs();
        if (numProcessors > 1)
            numProcessors -= 1;
        int maxDesiredProcessors = (totalNumWeights + 99999) / 100000;
        if (numProcessors > maxDesiredProcessors)
            numProcessors = maxDesiredProcessors;
        assert(numProcessors >= 1);
        omp_set_num_threads(numProcessors);
        uint64_t tensorToInitializePerThread = (totalNumWeights + (numProcessors - 1)) / numProcessors;
#pragma omp parallel
        {
            int threadNum = omp_get_thread_num();
            std::default_random_engine generator(time(NULL) * threadNumHash(threadNum));
            uint64_t threadEnd = (threadNum + 1) * tensorToInitializePerThread;
            if (totalNumWeights < threadEnd)
                threadEnd = totalNumWeights;
            for (uint64_t i = threadNum * tensorToInitializePerThread; i < threadEnd; ++i) {
                bufferMem[i] = (half)distribution(generator);
            }
        }

        performCopy(buffer, tensorToInitialize, streams, synchronous);
    }

    virtual shared_ptr<Initializer> clone() { return make_shared<UniformRandomInitializer>(*this); }

   protected:
    double maxValue;
    double minValue;
};

}  // namespace ThorImplementation
