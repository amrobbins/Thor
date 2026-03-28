#pragma once

#include <cstdint>

namespace Thor {

class Parameterizable {
public:
    virtual ~Parameterizable();

    Parameterizable(uint32_t id) : id(id) {}

    uint64_t getId() { return id; }

    private:
    const uint64_t id;
};

}
