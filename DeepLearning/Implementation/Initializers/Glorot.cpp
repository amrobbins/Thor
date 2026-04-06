#include "Glorot.h"

using namespace std;

namespace ThorImplementation {

Glorot::Glorot(Mode mode) : mode(mode) { assert(mode == Mode::UNIFORM || mode == Mode::NORMAL); }

Event Glorot::initialize() {
    if (mode == Mode::UNIFORM) {
        return initializeUniform();
    } else {
        return initializeNormal();
    }
}

Event Glorot::initializeUniform() {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    const uint32_t weightsGpuNum = weights.getPlacement().getDeviceNum();
    Tensor buffer = weights.clone(cpuPlacement);
 Stream initStream = stream.isPresent() ? stream.get() : Stream::getNextGradientUpdateStream(weightsGpuNum);

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
    omp_set_dynamic(0);
#pragma omp parallel num_threads(numProcessors)
    {
        int threadNum = omp_get_thread_num();
        const uint64_t start = uint64_t(threadNum) * chunk;
        const uint64_t end = min<uint64_t>(totalNumWeights, start + chunk);
        uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        using clock = chrono::high_resolution_clock;
        const uint64_t nanoseconds = chrono::duration_cast<chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        mt19937 generator(Tensor::getThreadIdHash64(nanoseconds));
        const float fanInOutTerm = static_cast<float>(sqrt(6.0 / static_cast<double>(layerFanIn + layerFanOut)));
        if (buffer.getDataType() == TensorDescriptor::DataType::FP16) {
            half *bufferMem = (half *)buffer.getMemPtr();
            for (uint64_t i = start; i < end; ++i) {
                float d = distribution(generator);
                float value = d * fanInOutTerm;
                bufferMem[i] = (half)value;
            }
        } else if (buffer.getDataType() == TensorDescriptor::DataType::FP32) {
            float *bufferMem = (float *)buffer.getMemPtr();
            for (uint64_t i = start; i < end; ++i) {
                float value = distribution(generator) * fanInOutTerm;
                bufferMem[i] = value;
            }
        } else {
            assert(false);
        }
    }

    weights.copyFromAsync(buffer, initStream);
    Event tensorInitializedEvent = initStream.putEvent();
    return tensorInitializedEvent;
}

Event Glorot::initializeNormal() {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    const uint32_t weightsGpuNum = weights.getPlacement().getDeviceNum();
    Tensor buffer = weights.clone(cpuPlacement);

    Stream initStream = stream.isPresent() ? stream.get() : Stream::getNextGradientUpdateStream(weightsGpuNum);

    float mean = 0.0;
    double variance = 2.0 / static_cast<double>(layerFanIn + layerFanOut);
    float standardDeviation = static_cast<float>(sqrt(variance));

    uint64_t totalNumWeights = weights.getDescriptor().getTotalNumElements();
    uint64_t numProcessors = omp_get_num_procs();
    if (numProcessors > 1)
        numProcessors -= 1;
    uint64_t maxDesiredProcessors = (totalNumWeights + 99999) / 100000;
    if (numProcessors > maxDesiredProcessors)
        numProcessors = maxDesiredProcessors;
    assert(numProcessors >= 1);
    const uint64_t chunk = (totalNumWeights + (numProcessors - 1)) / numProcessors;
    omp_set_dynamic(0);
#pragma omp parallel num_threads(numProcessors)
    {
        int threadNum = omp_get_thread_num();
        const uint64_t start = uint64_t(threadNum) * chunk;
        const uint64_t end = min<uint64_t>(totalNumWeights, start + chunk);
        normal_distribution<float> distribution(mean, standardDeviation);
        using clock = chrono::high_resolution_clock;
        const uint64_t nanoseconds = chrono::duration_cast<chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        mt19937 generator(Tensor::getThreadIdHash64(nanoseconds));
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

shared_ptr<Initializer> Glorot::clone() { return make_shared<Glorot>(*this); }

}  // namespace ThorImplementation
