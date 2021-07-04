#include "Glorot.h"

namespace ThorImplementation {

Glorot::Glorot(Mode mode) : mode(mode) { assert(mode == Mode::UNIFORM || mode == Mode::NORMAL); }

void Glorot::initialize(Layer *layer, Tensor tensorToInitialize) { Initializer::initialize(layer, tensorToInitialize); }

void Glorot::initialize(Layer *layer, Tensor tensorToInitialize, vector<Stream> streams) {
    if (mode == Mode::UNIFORM) {
        initializeUniform(layer->getFanIn(), layer->getFanOut(), tensorToInitialize, streams);
    } else {
        initializeNormal(layer->getFanIn(), layer->getFanOut(), tensorToInitialize, streams);
    }
}

void Glorot::initializeUniform(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, vector<Stream> streams) {
    std::hash<int> threadNumHash;
    Tensor buffer = tensorToInitialize.clone(TensorPlacement::MemDevices::CPU);

    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

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
            double value = distribution(generator) * sqrt(6.0 / (fanIn + fanOut));
            bufferMem[i] = (half)value;
        }
    }

    performCopy(buffer, tensorToInitialize, streams);
}

void Glorot::initializeNormal(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, vector<Stream> streams) {
    std::hash<int> threadNumHash;
    Tensor buffer = tensorToInitialize.clone(TensorPlacement::MemDevices::CPU);

    float mean = 0.0;
    double variance = 2.0 / (fanIn + fanOut);
    float standardDeviation = sqrt(variance);
    std::normal_distribution<float> distribution(mean, standardDeviation);

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

    performCopy(buffer, tensorToInitialize, streams);
}

shared_ptr<Initializer> Glorot::clone() { return make_shared<Glorot>(*this); }

}  // namespace ThorImplementation
