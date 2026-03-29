#pragma once

#include <atomic>
#include <cstdint>
#include <debug/unordered_map>

namespace Thor {

class Parameterizable {
   public:
    virtual ~Parameterizable() = default;

    Parameterizable();
    Parameterizable(uint64_t originalId);

    uint64_t getId() const { return id; }

   private:
    uint64_t id;
    uint64_t originalId;
    static std::atomic<int64_t> nextId;

    static std::mutex originalIdMapLock;
    static std::unordered_map<uint64_t, uint64_t> orignalIdToId;
};

}  // namespace Thor
