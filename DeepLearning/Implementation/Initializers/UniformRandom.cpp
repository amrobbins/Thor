#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

using namespace std;

namespace ThorImplementation {

UniformRandom::UniformRandom(double maxValue, double minValue) : maxValue(maxValue), minValue(minValue) {
    assert(isfinite(maxValue));
    assert(isfinite(minValue));
    assert(maxValue >= minValue);
}

Event UniformRandom::initialize(Layer *layer, Tensor tensorToInitialize) { return Initializer::initialize(layer, tensorToInitialize); }

Event UniformRandom::initialize(Layer *layer, Tensor tensorToInitialize, vector<Stream> streams) {
    // FIXME: I hard-coded half values here, so any uniformRandom init will only work for half type weights - networks will not train when not fp16
    Tensor buffer = tensorToInitialize.clone(TensorPlacement::MemDevices::CPU);

    bool constant = minValue == maxValue;
    half constantValue = half(minValue);

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
        uniform_real_distribution<float> distribution;
        if (!constant)
            distribution = uniform_real_distribution<float>(minValue, maxValue);
        using clock = chrono::high_resolution_clock;
        const uint64_t nanoseconds = chrono::duration_cast<chrono::nanoseconds>(
                                clock::now().time_since_epoch()).count();
        default_random_engine generator(Tensor::getThreadIdHash64(nanoseconds));
        uint64_t threadEnd = (threadNum + 1) * tensorToInitializePerThread;
        if (totalNumWeights < threadEnd)
            threadEnd = totalNumWeights;
        if (constant) {
            // Explicitly handle constant case:
            uint64_t start = threadNum * tensorToInitializePerThread;
            uint64_t count = threadEnd - start;
            if (constantValue == half(0)) {
                // Fastest way to fill zeros:
                memset(bufferMem + start, 0, count * sizeof(half));
            } else {
                fill_n(bufferMem + start, count, constantValue);
            }
        } else {
            for (uint64_t i = threadNum * tensorToInitializePerThread; i < threadEnd; ++i) {
                bufferMem[i] = (half)distribution(generator);
            }
        }
    }

    return performCopy(buffer, tensorToInitialize, streams);
}

shared_ptr<Initializer> UniformRandom::clone() { return make_shared<UniformRandom>(*this); }

}  // namespace ThorImplementation
