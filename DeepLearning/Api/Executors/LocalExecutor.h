#pragma once

#include "DeepLearning/Api/Executors/ExecutorBase.h"
#include "DeepLearning/Api/HyperparameterControllers/HyperparameterController.h"
#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"

#include <string>

namespace Thor {

class Executor;

using std::string;

class LocalExecutor : public ExecutorBase {
   public:
    class Builder;

    LocalExecutor() { initialized = false; }

    virtual ~LocalExecutor() {}

    void trainEpochs(double epochs);
    void trainBatches(uint32_t batches);
    void createSnapshot(string filepath);

   private:
    bool initialized;

    Network network;
    Loader loader;
    HyperparameterController hyperparameterController;
    vector<Visualizer> visualizers;
};

class LocalExecutor::Builder {
   public:
    virtual Executor build();

    LocalExecutor::Builder network(Network _network) {
        assert(!this->_network.isPresent());
        this->_network = _network;
        return *this;
    }

    LocalExecutor::Builder loader(Loader _loader) {
        assert(!this->_loader.isPresent());
        this->_loader = _loader;
        return *this;
    }

    LocalExecutor::Builder hyperparameterController(HyperparameterController _hyperparameterController) {
        assert(!this->_hyperparameterController.isPresent());
        this->_hyperparameterController = _hyperparameterController;
        return *this;
    }

    LocalExecutor::Builder visualizer(Visualizer _visualizer) {
        if (_visualizers.isEmpty())
            _visualizers = vector<Visualizer>();
        _visualizers.get().push_back(_visualizer);
        return *this;
    }

   private:
    Optional<Network> _network;
    Optional<Loader> _loader;
    Optional<HyperparameterController> _hyperparameterController;
    Optional<vector<Visualizer>> _visualizers;
};

}  // namespace Thor