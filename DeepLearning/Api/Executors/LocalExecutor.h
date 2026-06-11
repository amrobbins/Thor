#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Executors/Executor.h"
#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Training/Observers/LineStatsReporter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cuda_profiler_api.h>
#include <filesystem>
#include <fstream>

#include <condition_variable>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <utility>

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

    ~LocalExecutor() override;

    // FIXME: need train, validate and test and no exampleType
    void trainEpochs(uint32_t numEpochs, std::set<std::string> tensorsToReturn);
    void createSnapshot(std::string filepath) {}  // FIXME

    bool isBatchDataReady();
    void waitForBatchData();
    std::unordered_map<std::string, std::vector<uint8_t>> popBatchData();

   private:
    bool initialized;

    std::shared_ptr<PlacedNetwork> placedNetwork;
    std::shared_ptr<Loader> loader;
    std::shared_ptr<Optimizer> optimizer;
    std::vector<std::shared_ptr<TrainingObserver>> observers;
    bool statsEnabled = true;
    double statsIntervalSeconds = 10.0;

    //    std::vector<ThorImplementation::StampedNetwork> stampedNetworks;

    std::shared_ptr<std::mutex> epochMutex;
    std::shared_ptr<uint64_t> currentEpoch;
    std::shared_ptr<uint64_t> numBatchesDoneInEpoch;
    std::shared_ptr<uint64_t> numBatchesInEpoch;
    std::shared_ptr<std::condition_variable> batchFinished;
    std::condition_variable batchDataPopped;
    std::shared_ptr<std::map<uint64_t, bool>> batchDataReady;
    std::shared_ptr<std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<uint8_t>>>> batchData;

    std::string outputDirectory;

    // stampNumber -> [ (start0, finish0), (start1, finish1), ... ]
    std::unordered_map<uint64_t, std::deque<std::pair<Event, Event>>> batchletTimingEvents;

    bool isBatchDataReadyUnlocked();
    void waitForBatchDataUnlocked(std::unique_lock<std::mutex>& lck);

    void trainBatches(uint64_t initialEpochBatchNum, uint64_t batches, ExampleType exampleType, std::set<std::string> tensorsToReturn);
    void emitTrainingEvent(const TrainingEvent& event);

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
        THOR_THROW_IF_FALSE(_loader);
        THOR_THROW_IF_FALSE(!this->_loader);
        this->_loader = _loader;
        return *this;
    }

    LocalExecutor::Builder optimizer(std::shared_ptr<Optimizer> _optimizer) {
        THOR_THROW_IF_FALSE(_optimizer);
        THOR_THROW_IF_FALSE(!this->_optimizer);
        this->_optimizer = _optimizer;
        return *this;
    }

    LocalExecutor::Builder observer(std::shared_ptr<TrainingObserver> observer) {
        THOR_THROW_IF_FALSE(observer);
        if (!observers_.has_value()) {
            observers_ = std::vector<std::shared_ptr<TrainingObserver>>();
        }
        observers_.value().push_back(std::move(observer));
        return *this;
    }

    LocalExecutor::Builder statsEnabled(bool statsEnabled) {
        statsEnabled_ = statsEnabled;
        return *this;
    }

    LocalExecutor::Builder statsIntervalSeconds(double statsIntervalSeconds) {
        THOR_THROW_IF_FALSE(statsIntervalSeconds >= 0.0);
        statsIntervalSeconds_ = statsIntervalSeconds;
        return *this;
    }

    LocalExecutor::Builder outputDirectory(std::string _outputDirectory) {
        THOR_THROW_IF_FALSE(!this->_outputDirectory.has_value());
        if (_outputDirectory.empty())
            _outputDirectory = "./";
        std::filesystem::path outputPath = std::filesystem::absolute(std::filesystem::path(_outputDirectory));
        this->_outputDirectory = std::filesystem::canonical(outputPath).string();
        return *this;
    }

   private:
    Network* _network;
    std::shared_ptr<Loader> _loader;
    std::shared_ptr<Optimizer> _optimizer;
    std::optional<std::vector<std::shared_ptr<TrainingObserver>>> observers_;
    bool statsEnabled_ = true;
    double statsIntervalSeconds_ = 10.0;
    std::optional<std::string> _outputDirectory;
};

}  // namespace Thor
