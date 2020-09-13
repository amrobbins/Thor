#pragma once

#include "DeepLearning/Api/Executors/ExecutorBase.h"

namespace Thor {

class LocalExecutor : public ExecutorBase {
   public:
    virtual ~LocalExecutor();
};

}  // namespace Thor
