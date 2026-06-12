#include "DeepLearning/Api/Training/Executors/LocalExecutorTrainingRunner.h"

#include "DeepLearning/Api/Executors/LocalExecutor.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <set>
#include <stdexcept>

namespace Thor {

namespace {

class BorrowedTrainingObserver : public TrainingObserver {
   public:
    explicit BorrowedTrainingObserver(TrainingObserver& observer) : observer(observer) {}
    void onTrainingEvent(const TrainingEvent& event) override { observer.onTrainingEvent(event); }

   private:
    TrainingObserver& observer;
};

}  // namespace

void runLocalExecutorBackedTraining(const TrainingRunRequest& request,
                                    TrainingObserver& observer,
                                    const LocalExecutorTrainingOptions& options) {
    THOR_THROW_IF_FALSE(request.network != nullptr);
    THOR_THROW_IF_FALSE(request.loader != nullptr);
    THOR_THROW_IF_FALSE(request.epochs > 0);
    THOR_THROW_IF_FALSE(options.maxInFlightBatches >= 1);

    if (request.trainingProgram.has_value() && !request.trainingProgram->isInitialized()) {
        throw std::runtime_error("Trainer execution received an uninitialized TrainingProgram.");
    }
    auto borrowedObserver = std::make_shared<BorrowedTrainingObserver>(observer);
    LocalExecutor::Builder builder = LocalExecutor::Builder()
                                         .network(*request.network)
                                         .loader(request.loader)
                                         .trainingProgram(request.trainingProgram)
                                         .observer(borrowedObserver)
                                         .statsEnabled(request.runtime.statsEnabled)
                                         .statsIntervalSeconds(request.runtime.statsIntervalSeconds)
                                         .maxInFlightBatches(options.maxInFlightBatches)
                                         .synchronizeAfterEveryBatch(options.synchronizeAfterEveryBatch);
    if (request.optimizer != nullptr) {
        builder = builder.optimizer(request.optimizer);
    }
    std::shared_ptr<LocalExecutor> localExecutor = builder.build();
    localExecutor->trainEpochs(request.epochs, request.runtime.statsEnabled ? request.runtime.scalarTensorsToReport
                                                                             : std::set<std::string>{});
}

}  // namespace Thor
