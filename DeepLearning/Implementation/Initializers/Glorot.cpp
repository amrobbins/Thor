#include "Glorot.h"

using namespace std;

namespace ThorImplementation {

Glorot::Glorot(Mode mode) : mode(mode) { assert(mode == Mode::UNIFORM || mode == Mode::NORMAL); }

Event Glorot::initialize(Layer *layer, Tensor tensorToInitialize) { return Initializer::initialize(layer, tensorToInitialize); }

Event Glorot::initialize(Layer *layer, Tensor tensorToInitialize, vector<Stream> streams) {
    if (mode == Mode::UNIFORM) {
        return initializeUniform(layer->getFanIn(), layer->getFanOut(), tensorToInitialize, streams);
    } else {
        return initializeNormal(layer->getFanIn(), layer->getFanOut(), tensorToInitialize, streams);
    }
}

Event Glorot::initializeUniform(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, vector<Stream> streams) {
    // FIXME: I hard-coded half values here, so any uniformRandom init will only work for half type weights - networks will not train when
    // not fp16
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
        const uint64_t start = uint64_t(threadNum) * chunk;
        const uint64_t end = min<uint64_t>(totalNumWeights, start + chunk);
        uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        using clock = chrono::high_resolution_clock;
        const uint64_t nanoseconds = chrono::duration_cast<chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        mt19937 generator(Tensor::getThreadIdHash64(nanoseconds));
        for (uint64_t i = start; i < end; ++i) {
            double value = distribution(generator) * sqrt(6.0 / (fanIn + fanOut));
            bufferMem[i] = (half)value;
        }
    }

    return performCopy(buffer, tensorToInitialize, streams);
}

Event Glorot::initializeNormal(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, vector<Stream> streams) {
    // FIXME: I hard-coded half values here, so any uniformRandom init will only work for half type weights - networks will not train when
    // not fp16
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor buffer = tensorToInitialize.clone(cpuPlacement);

    float mean = 0.0;
    double variance = 2.0 / (fanIn + fanOut);
    float standardDeviation = sqrt(variance);

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
        const uint64_t start = uint64_t(threadNum) * chunk;
        const uint64_t end = min<uint64_t>(totalNumWeights, start + chunk);
        normal_distribution<float> distribution(mean, standardDeviation);
        using clock = chrono::high_resolution_clock;
        const uint64_t nanoseconds = chrono::duration_cast<chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        mt19937 generator(Tensor::getThreadIdHash64(nanoseconds));
        for (uint64_t i = start; i < end; ++i) {
            bufferMem[i] = (half)distribution(generator);
        }
    }

    return performCopy(buffer, tensorToInitialize, streams);
}

shared_ptr<Initializer> Glorot::clone() { return make_shared<Glorot>(*this); }

}  // namespace ThorImplementation
