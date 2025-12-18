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
    // FIXME: I hard-coded half values here, so any uniformRandom init will only work for half type weights - networks will not train when
    // not fp16
    bool constant = minValue == maxValue;
    if (constant) {
        tensorToInitialize.fill(minValue, streams[0]);
        return streams[0].putEvent();
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor buffer = tensorToInitialize.clone(cpuPlacement);
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
    const uint64_t chunk = (totalNumWeights + (numProcessors - 1)) / numProcessors;
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        uniform_real_distribution<float> distribution;
        if (!constant)
            distribution = uniform_real_distribution<float>(minValue, maxValue);
        using clock = chrono::high_resolution_clock;
        const uint64_t nanoseconds = chrono::duration_cast<chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        mt19937 generator(Tensor::getThreadIdHash64(nanoseconds));
        const uint64_t start = uint64_t(threadNum) * chunk;
        const uint64_t end = min<uint64_t>(totalNumWeights, start + chunk);
        for (uint64_t i = start; i < end; ++i) {
            bufferMem[i] = (half)distribution(generator);
        }
    }

    return performCopy(buffer, tensorToInitialize, streams);
}

shared_ptr<Initializer> UniformRandom::clone() { return make_shared<UniformRandom>(*this); }

}  // namespace ThorImplementation
