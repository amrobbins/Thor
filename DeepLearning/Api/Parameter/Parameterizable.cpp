#include "DeepLearning/Api/Parameter/Parameterizable.h"

using namespace std;

namespace Thor {

atomic<int64_t> Parameterizable::nextId(2);
unordered_map<uint64_t, uint64_t> Parameterizable::orignalIdToId;
std::mutex Parameterizable::originalIdMapLock;

Parameterizable::Parameterizable() : id(nextId.fetch_add(1)) { originalId = id; }

Parameterizable::Parameterizable(uint64_t originalId) {
    lock_guard<mutex> lock(originalIdMapLock);
    this->originalId = originalId;
    if (orignalIdToId.contains(originalId)) {
        id = orignalIdToId[originalId];
    } else {
        id = nextId.fetch_add(1);
        orignalIdToId[originalId] = id;
    }
}

}  // namespace Thor
