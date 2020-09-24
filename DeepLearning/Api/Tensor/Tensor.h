#pragma once

#include <assert.h>
#include <atomic>
#include <utility>
#include <vector>

using std::atomic;
using std::vector;

namespace Thor {

class Tensor {
   public:
    enum class DataType { FP32 = 2, FP16 };

    Tensor() : initialized(false) {}
    Tensor(DataType dataType, vector<uint64_t> dimensions)
        : id(nextId.fetch_add(1)), dataType(dataType), dimensions(dimensions), initialized(true) {}
    virtual ~Tensor() {}

    // Cloned tensors have identical characteristics but different id's
    Tensor clone() {
        Tensor cloned;
        cloned.dataType = dataType;
        cloned.dimensions = dimensions;
        cloned.initialized = initialized;
        return cloned;
    }

    uint64_t getId() const {
        assert(initialized);
        return id;
    }
    DataType getDataType() {
        assert(initialized);
        return dataType;
    }
    vector<uint64_t> getDimensions() {
        assert(initialized);
        return dimensions;
    }

    bool isInitialized() { return initialized; }

    bool operator==(const Tensor &other) const { return id == other.id; }
    bool operator!=(const Tensor &other) const { return id != other.id; }
    bool operator<(const Tensor &other) const { return id < other.id; }
    bool operator>(const Tensor &other) const { return id > other.id; }

    static bool dataTypeValid(DataType dataType) { return dataType == DataType::FP32 || dataType == DataType::FP16; }

   private:
    uint64_t id;
    static atomic<uint64_t> nextId;

    DataType dataType;
    vector<uint64_t> dimensions;

    bool initialized;
};

}  // namespace Thor
