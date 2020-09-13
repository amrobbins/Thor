#pragma once

#include "DeepLearning/Api/Executors/AwsExecutor.h"
#include "DeepLearning/Api/Executors/ExecutorBase.h"
#include "DeepLearning/Api/Executors/LocalExecutor.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Executor {
   public:
    Executor() {}
    Executor(ExecutorBase *executorBase);

    virtual ~Executor() {}

    Executor *getExecutor();

   private:
    shared_ptr<ExecutorBase> executor;
};

}  // namespace Thor
