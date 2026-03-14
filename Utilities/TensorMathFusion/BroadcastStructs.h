#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ThorImplementation {

struct BroadcastInfoHeader {
    uint32_t rank;
    uint32_t num_inputs;
    uint64_t numel;
};

class BroadcastInfoHostBuffer {
   public:
    BroadcastInfoHostBuffer(uint32_t rank, uint32_t num_inputs, uint64_t numel);

    static size_t bytesRequired(uint32_t rank, uint32_t num_inputs);

    BroadcastInfoHeader* header();
    const BroadcastInfoHeader* header() const;

    uint64_t* outputStrides();
    const uint64_t* outputStrides() const;

    uint64_t* inputStrides();
    const uint64_t* inputStrides() const;

    uint64_t& inputStride(uint32_t inputIndex, uint32_t dim);
    const uint64_t& inputStride(uint32_t inputIndex, uint32_t dim) const;

    void* data();
    const void* data() const;
    size_t sizeBytes() const;

   private:
    std::vector<std::byte> storage;
};

}  // namespace ThorImplementation
