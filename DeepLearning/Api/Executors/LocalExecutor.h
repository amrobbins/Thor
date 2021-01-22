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

struct LoanedBufferMetadata {
    LoanedBufferMetadata() {}
    LoanedBufferMetadata(Event doneWithBufferEvent, ExampleType exampleType, map<std::string, ThorImplementation::Tensor> batchTensorMap) {
        this->doneWithBufferEvent = doneWithBufferEvent;
        this->exampleType = exampleType;
        this->batchTensorMap = batchTensorMap;
    }

    Event doneWithBufferEvent;
    ExampleType exampleType;
    std::map<std::string, ThorImplementation::Tensor> batchTensorMap;
};

class LocalExecutor : public Executor {
   public:
    class Builder;

    LocalExecutor() { initialized = false; }

    virtual ~LocalExecutor() {}

    // FIXME: need train, validate and test and no exampleType
    void trainTillEpochIsFinished(ExampleType exampleType);
    uint64_t trainBatches(uint32_t batches, ExampleType exampleType);
    void createSnapshot(std::string filepath) {}  // FIXME

   private:
    bool initialized;

    Network network;
    std::shared_ptr<Loader> loader;
    HyperparameterController hyperparameterController;
    vector<std::shared_ptr<Visualizer>> visualizers;

    vector<ThorImplementation::StampedNetwork> stampedNetworks;

    unique_ptr<AsyncQueue<ExecutionState>> hyperparameterControllerExecutionState;
    vector<unique_ptr<AsyncQueue<ExecutionState>>> visualizerExecutionState;

    unique_ptr<AsyncQueue<LoanedBufferMetadata>> loanedBufferQueue;
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

    LocalExecutor::Builder visualizer(std::shared_ptr<Visualizer> _visualizer) {
        if (_visualizers.isEmpty())
            _visualizers = vector<shared_ptr<Visualizer>>();
        _visualizers.get().push_back(_visualizer);
        return *this;
    }

   private:
    Optional<Network> _network;
    std::shared_ptr<Loader> _loader;
    Optional<HyperparameterController> _hyperparameterController;
    Optional<vector<std::shared_ptr<Visualizer>>> _visualizers;
};

}  // namespace Thor
