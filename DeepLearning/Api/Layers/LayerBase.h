#pragma once

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class LayerBase {
   public:
    LayerBase() : id(nextId.fetch_add(1)) {}
    virtual ~LayerBase() {}

    bool operator==(const LayerBase &other) const { return id == other.id; }
    bool operator!=(const LayerBase &other) const { return id != other.id; }
    bool operator<(const LayerBase &other) const { return id < other.id; }
    bool operator>(const LayerBase &other) const { return id > other.id; }

   private:
    const uint32_t id;
    static atomic<uint32_t> nextId;
};

}  // namespace Thor
