#pragma once

#include <atomic>
#include <cstdint>
#include <debug/unordered_map>

namespace Thor {

class Parameterizable {
   public:
    virtual ~Parameterizable() = default;

    Parameterizable() = default;
};

}  // namespace Thor
