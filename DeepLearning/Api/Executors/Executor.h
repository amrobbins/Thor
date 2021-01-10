#pragma once

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Executor {
   public:
    Executor() {}

    virtual ~Executor() {}
};

}  // namespace Thor
