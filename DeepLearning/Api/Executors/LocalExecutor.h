#pragma once

#include "DeepLearning/Api/Executors/ExecutionState.h"
#include "DeepLearning/Api/Executors/Executor.h"
#include "DeepLearning/Api/HyperparameterControllers/HyperparameterController.h"
#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <map>
#include <memory>
#include <string>

namespace Thor {

class Executor;

struct BufferStampTensorsParams {
    std::shared_ptr<std::unordered_map<uint64_t, std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>>> batchletData;
    std::shared_ptr<std::unordered_map<uint64_t, std::unordered_map<std::string, std::vector<uint8_t>>>> batchData;
    std::shared_ptr<mutex> mtx;
    std::shared_ptr<Loader> loader;

    ExampleType exampleType;
    uint64_t epochBatchNum;
    uint64_t numBatchletsInBatch;

    set<string> tensorsToReturn;
    map<string, ThorImplementation::Tensor> batchletInput;
    map<string, ThorImplementation::Tensor> batchletOutput;
};

class LocalExecutor : public Executor {
   public:
    class Builder;

    LocalExecutor() { initialized = false; }

    virtual ~LocalExecutor() {}

    // FIXME: need train, validate and test and no exampleType
    void trainTillEpochIsFinished(ExampleType exampleType, set<string> tensorsToReturn);
    uint64_t trainBatches(uint64_t batches, ExampleType exampleType, set<string> tensorsToReturn);
    void createSnapshot(std::string filepath) {}  // FIXME

   private:
    bool initialized;

    Network network;
    std::shared_ptr<Loader> loader;
    HyperparameterController hyperparameterController;
    vector<Visualizer*> visualizers;

    vector<ThorImplementation::StampedNetwork> stampedNetworks;

    unique_ptr<AsyncQueue<ExecutionState>> hyperparameterControllerExecutionState;
    vector<unique_ptr<AsyncQueue<ExecutionState>>> visualizerExecutionState;

    // stampNumber -> [ (start0, finish0), (start1, finish1), ... ]
    std::unordered_map<uint64_t, std::deque<pair<Event, Event>>> batchletTimingEvents;

    static void CUDART_CB bufferStampTensors(void* data);
};

class LocalExecutor::Builder {
   public:
    virtual std::shared_ptr<LocalExecutor> build();

    LocalExecutor::Builder network(Network _network) {
        assert(!this->_network.isPresent());
        this->_network = _network;
        return *this;
    }

    LocalExecutor::Builder loader(std::shared_ptr<Loader> _loader) {
        assert(_loader);
        assert(!this->_loader);
        this->_loader = _loader;
        return *this;
    }

    LocalExecutor::Builder hyperparameterController(HyperparameterController _hyperparameterController) {
        assert(!this->_hyperparameterController.isPresent());
        this->_hyperparameterController = _hyperparameterController;
        return *this;
    }

    LocalExecutor::Builder visualizer(Visualizer* _visualizer) {
        if (_visualizers.isEmpty())
            _visualizers = vector<Visualizer*>();
        _visualizers.get().push_back(_visualizer);
        return *this;
    }

   private:
    Optional<Network> _network;
    std::shared_ptr<Loader> _loader;
    Optional<HyperparameterController> _hyperparameterController;
    Optional<vector<Visualizer*>> _visualizers;
};

}  // namespace Thor
