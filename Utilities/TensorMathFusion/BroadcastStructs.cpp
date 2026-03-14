// FIXME: DELETE
// #include "Utilities/TensorMathFusion/BroadcastStructs.h"
//
// #include <stdexcept>
//
// using namespace ThorImplementation;
//
// BroadcastInfoBufferView::BroadcastInfoBufferView(uint32_t rank, uint32_t num_inputs, uint64_t numel)
//     : storage(bytesRequired(rank, num_inputs), std::byte{0}) {
//     BroadcastInfoHeader* h = header();
//     h->rank = rank;
//     h->num_inputs = num_inputs;
//     h->numel = numel;
// }
//
// size_t BroadcastInfoHostBuffer::bytesRequired(uint32_t rank, uint32_t num_inputs) {
//     return sizeof(BroadcastInfoHeader) + sizeof(uint64_t) * rank + sizeof(uint64_t) * rank * num_inputs;
// }
//
// BroadcastInfoHeader* BroadcastInfoHostBuffer::header() { return reinterpret_cast<BroadcastInfoHeader*>(storage.data()); }
//
// const BroadcastInfoHeader* BroadcastInfoHostBuffer::header() const { return reinterpret_cast<const BroadcastInfoHeader*>(storage.data());
// }
//
// uint64_t* BroadcastInfoHostBuffer::outputStrides() { return reinterpret_cast<uint64_t*>(storage.data() + sizeof(BroadcastInfoHeader)); }
//
// const uint64_t* BroadcastInfoHostBuffer::outputStrides() const {
//     return reinterpret_cast<const uint64_t*>(storage.data() + sizeof(BroadcastInfoHeader));
// }
//
// uint64_t* BroadcastInfoHostBuffer::inputStrides() { return outputStrides() + header()->rank; }
//
// const uint64_t* BroadcastInfoHostBuffer::inputStrides() const { return outputStrides() + header()->rank; }
//
// uint64_t& BroadcastInfoHostBuffer::inputStride(uint32_t inputIndex, uint32_t dim) {
//     if (inputIndex >= header()->num_inputs)
//         throw std::out_of_range("BroadcastInfoHostBuffer::inputStride inputIndex out of range");
//     if (dim >= header()->rank)
//         throw std::out_of_range("BroadcastInfoHostBuffer::inputStride dim out of range");
//
//     return inputStrides()[static_cast<size_t>(inputIndex) * header()->rank + dim];
// }
//
// const uint64_t& BroadcastInfoHostBuffer::inputStride(uint32_t inputIndex, uint32_t dim) const {
//     if (inputIndex >= header()->num_inputs)
//         throw std::out_of_range("BroadcastInfoHostBuffer::inputStride inputIndex out of range");
//     if (dim >= header()->rank)
//         throw std::out_of_range("BroadcastInfoHostBuffer::inputStride dim out of range");
//
//     return inputStrides()[static_cast<size_t>(inputIndex) * header()->rank + dim];
// }
//
// void* BroadcastInfoHostBuffer::data() { return storage.data(); }
//
// const void* BroadcastInfoHostBuffer::data() const { return storage.data(); }
//
// size_t BroadcastInfoHostBuffer::sizeBytes() const { return storage.size(); }
