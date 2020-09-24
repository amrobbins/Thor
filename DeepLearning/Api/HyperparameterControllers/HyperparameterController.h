#pragma once

#include "DeepLearning/Api/HyperparameterControllers/HyperparameterControllerBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class HyperparameterController {
   public:
    HyperparameterController() {}
    HyperparameterController(HyperparameterControllerBase *hyperparameterControllerBase);

    virtual ~HyperparameterController() {}

    HyperparameterController *getHyperparameterController();

   private:
    shared_ptr<HyperparameterControllerBase> hyperparameterController;
};

}  // namespace Thor
