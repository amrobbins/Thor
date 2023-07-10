#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/Executors/Executor.h"
#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cuda_profiler_api.h>
#include <boost/filesystem.hpp>

#include <condition_variable>
#include <map>
#include <memory>
#include <string>
#include <thread>

namespace Thor {

class Executor;

struct BufferStampTensorsParams {
    std::shared_ptr<std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>> batchletData;
    std::shared_ptr<std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<uint8_t>>>> batchData;
    std::shared_ptr<std::mutex> batchMutex;
    std::shared_ptr<std::map<uint64_t, bool>> batchDataReady;
    std::shared_ptr<std::mutex> epochMutex;
    std::shared_ptr<uint64_t> numBatchesDoneInEpoch;
    std::shared_ptr<std::condition_variable> batchFinished;
    std::shared_ptr<Loader> loader;

    ExampleType exampleType;
    uint64_t epochBatchNum;
    uint64_t numBatchletsInBatch;
    uint64_t numBatchesInEpoch;

    std::set<std::string> tensorsToReturn;
    std::map<std::string, ThorImplementation::Tensor> batchletInput;
    std::map<std::string, ThorImplementation::Tensor> batchletOutput;
};

class LocalExecutor : public Executor {
   public:
    class Builder;

    LocalExecutor() { initialized = false; }

    virtual ~LocalExecutor();

    // FIXME: need train, validate and test and no exampleType
    void trainEpochs(uint32_t numEpochs, std::set<std::string> tensorsToReturn);
    void createSnapshot(std::string filepath) {}  // FIXME

    bool isBatchDataReady();
    void waitForBatchData();
    std::unordered_map<std::string, std::vector<uint8_t>> popBatchData();

   private:
    bool initialized;

    Network* network;
    std::shared_ptr<Loader> loader;
    std::shared_ptr<Optimizer> optimizer;
    // FIXME: shared_ptr, however how do I deal with singletons then?
    std::vector<Visualizer*> visualizers;

    std::vector<ThorImplementation::StampedNetwork> stampedNetworks;

    std::shared_ptr<std::mutex> epochMutex;
    std::shared_ptr<uint64_t> currentEpoch;
    std::shared_ptr<uint64_t> numBatchesDoneInEpoch;
    std::shared_ptr<uint64_t> numBatchesInEpoch;
    std::shared_ptr<std::condition_variable> batchFinished;
    std::condition_variable batchDataPopped;
    std::shared_ptr<std::map<uint64_t, bool>> batchDataReady;
    std::shared_ptr<std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<uint8_t>>>> batchData;

    std::vector<std::shared_ptr<AsyncQueue<ExecutionState>>> visualizerExecutionState;

    std::string outputDirectory;

    // stampNumber -> [ (start0, finish0), (start1, finish1), ... ]
    std::unordered_map<uint64_t, std::deque<std::pair<Event, Event>>> batchletTimingEvents;

    bool isBatchDataReadyUnlocked();
    void waitForBatchDataUnlocked(std::unique_lock<std::mutex>& lck);

    void trainBatches(uint64_t initialEpochBatchNum, uint64_t batches, ExampleType exampleType, std::set<std::string> tensorsToReturn);

    static void CUDART_CB bufferStampTensors(void* data);
};

class LocalExecutor::Builder {
   public:
    virtual std::shared_ptr<LocalExecutor> build();

    LocalExecutor::Builder network(Network& _network) {
        this->_network = &_network;
        return *this;
    }

    LocalExecutor::Builder loader(std::shared_ptr<Loader> _loader) {
        assert(_loader);
        assert(!this->_loader);
        this->_loader = _loader;
        return *this;
    }

    LocalExecutor::Builder optimizer(std::shared_ptr<Optimizer> _optimizer) {
        assert(_optimizer);
        assert(!this->_optimizer);
        this->_optimizer = _optimizer;
        return *this;
    }

    LocalExecutor::Builder visualizer(Visualizer* _visualizer) {
        if (_visualizers.isEmpty())
            _visualizers = std::vector<Visualizer*>();
        _visualizers.get().push_back(_visualizer);
        return *this;
    }

    LocalExecutor::Builder outputDirectory(std::string _outputDirectory) {
        assert(this->_outputDirectory.isEmpty());
        if (_outputDirectory.empty())
            _outputDirectory = "./";
        boost::filesystem::path outputPath = boost::filesystem::absolute(boost::filesystem::path(_outputDirectory));
        this->_outputDirectory = boost::filesystem::canonical(outputPath).string();
        return *this;
    }

   private:
    Network* _network;
    std::shared_ptr<Loader> _loader;
    std::shared_ptr<Optimizer> _optimizer;
    Optional<std::vector<Visualizer*>> _visualizers;
    Optional<std::string> _outputDirectory;
};

}  // namespace Thor
