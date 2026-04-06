#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

using namespace std;

namespace ThorImplementation {

UniformRandom::UniformRandom(float maxValue, float minValue) : maxValue(maxValue), minValue(minValue) {
    assert(isfinite(maxValue));
    assert(isfinite(minValue));
    assert(maxValue >= minValue);
}

Event UniformRandom::initialize() {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    const uint32_t weightsGpuNum = weights.getPlacement().getDeviceNum();
    Stream initStream = stream.isPresent() ? stream.get() : Stream::getNextGradientUpdateStream(weightsGpuNum);
    Tensor buffer = weights.clone(cpuPlacement);

    bool constant = minValue == maxValue;
    if (constant) {
        weights.fill(minValue, initStream);
        return initStream.putEvent();
    }
 uint64_t totalNumWeights = weights.getDescriptor().getTotalNumElements();
    uint64_t numProcessors = omp_get_num_procs();
    if (numProcessors > 1)
        numProcessors -= 1;
    uint64_t maxDesiredProcessors = (totalNumWeights + 99999) / 100000;
    if (numProcessors > maxDesiredProcessors)
        numProcessors = maxDesiredProcessors;
    assert(numProcessors >= 1);
    omp_set_num_threads(static_cast<int>(numProcessors));
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

    weights.copyFromAsync(buffer, initStream);
    Event tensorInitializedEvent = initStream.putEvent();
    return tensorInitializedEvent;
}

shared_ptr<Initializer> UniformRandom::clone() { return make_shared<UniformRandom>(*this); }

}  // namespace ThorImplementation
