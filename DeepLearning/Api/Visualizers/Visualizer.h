#pragma once

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Visualizer {
   public:
    Visualizer() {}

    virtual ~Visualizer() {}
};

}  // namespace Thor
