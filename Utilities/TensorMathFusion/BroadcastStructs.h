#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ThorImplementation {

struct BroadcastInfoHeader {
    uint32_t rank;
    uint32_t _pad = 0;
    uint64_t numel;
};

class BroadcastInfoBufferView {
   public:
    BroadcastInfoBufferView(void* data, uint32_t rank, uint32_t num_inputs)
        : data_(static_cast<std::byte*>(data)), num_inputs_(num_inputs) {
        auto* h = header();
        h->rank = rank;
    }

    static size_t bytesRequired(uint32_t rank, uint32_t num_inputs) {
        return sizeof(BroadcastInfoHeader) + sizeof(uint64_t) * rank + sizeof(uint64_t) * rank * num_inputs;
    }

    BroadcastInfoHeader* header() { return reinterpret_cast<BroadcastInfoHeader*>(data_); }

    const BroadcastInfoHeader* header() const { return reinterpret_cast<const BroadcastInfoHeader*>(data_); }

    uint64_t* outputStrides() { return reinterpret_cast<uint64_t*>(data_ + sizeof(BroadcastInfoHeader)); }

    const uint64_t* outputStrides() const { return reinterpret_cast<const uint64_t*>(data_ + sizeof(BroadcastInfoHeader)); }

    uint64_t* inputStrides() { return outputStrides() + header()->rank; }

    const uint64_t* inputStrides() const { return outputStrides() + header()->rank; }

    uint64_t& inputStride(uint32_t inputIndex, uint32_t dim) {
        return inputStrides()[static_cast<size_t>(inputIndex) * header()->rank + dim];
    }

    const uint64_t& inputStride(uint32_t inputIndex, uint32_t dim) const {
        return inputStrides()[static_cast<size_t>(inputIndex) * header()->rank + dim];
    }

    size_t sizeBytes() const { return bytesRequired(header()->rank, num_inputs_); }

   private:
    std::byte* data_;
    uint32_t num_inputs_;
};

}  // namespace ThorImplementation
