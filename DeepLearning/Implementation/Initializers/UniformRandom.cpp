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
    // There are a few issues here NEED TO FIX THIS NEXT
    // 1. deserialization initialize branch does not actually call initializer.initialize, and the event is not present
    // 2. old initialization logic is still there, fighting with the incomplete new implementation, that is the place where initialize()
    //    is actually being called. See StampedNetwork.cpp line 46. All of this is causing the race issue in BatchNormalization test
    //    - surprising that I did not see this in other serialize tests, needs to be fixed everywhere weights are initialized.

    bool constant = minValue == maxValue;
    if (constant) {
        tensorToInitialize.fill(minValue, streams[0]);
        return streams[0].putEvent();
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor buffer = tensorToInitialize.clone(cpuPlacement);
    uint64_t totalNumWeights = tensorToInitialize.getDescriptor().getTotalNumElements();
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
        if (buffer.getDataType() == TensorDescriptor::DataType::FP16) {
            half *bufferMem = (half *)buffer.getMemPtr();
            for (uint64_t i = start; i < end; ++i) {
                bufferMem[i] = (half)distribution(generator);
            }
        } else if (buffer.getDataType() == TensorDescriptor::DataType::FP32) {
            float *bufferMem = (float *)buffer.getMemPtr();
            for (uint64_t i = start; i < end; ++i) {
                bufferMem[i] = distribution(generator);
            }
        } else {
            assert(false);
        }
    }

    return performCopy(buffer, tensorToInitialize, streams);
}

shared_ptr<Initializer> UniformRandom::clone() { return make_shared<UniformRandom>(*this); }

}  // namespace ThorImplementation
