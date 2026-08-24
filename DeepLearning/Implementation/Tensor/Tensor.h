#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"
#include "Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <assert.h>
#include <stdexcept>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <omp.h>

namespace ThorImplementation {

class RowPartitionRuntime;

/**
 * A multidimensional array allocated in either CPU or device memory.
 *
 * Tensor handles use std::shared_ptr-backed allocation ownership. Distinct
 * Tensor handle objects may be copied, assigned, reset, and destroyed
 * concurrently while sharing one allocation. Concurrent mutation of the same
 * Tensor handle object requires external synchronization, matching the
 * SharedOwnership.h contract. View metadata (descriptor, offset, and strides)
 * belongs to each Tensor handle while BackingMemory owns the allocation.
 */

class Tensor {
    friend class TypeConverter;
    friend class RowPartitionRuntime;

   public:
    Tensor() = default;
    Tensor(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes = 0);
    Tensor(const Tensor &tensorInstance) = default;
    Tensor(Tensor &&tensorInstance) noexcept = default;
    Tensor &operator=(const Tensor &tensorInstance) = default;
    Tensor &operator=(Tensor &&tensorInstance) noexcept = default;
    virtual ~Tensor() = default;

    bool isInitialized() const { return backingMemory != nullptr; }

    Tensor clone() const { return uninitialized() ? Tensor() : Tensor(placement, descriptor); }
    Tensor clone(TensorPlacement newPlacement) const { return uninitialized() ? Tensor() : Tensor(newPlacement, descriptor); }
    Tensor clone(DataType newDataType) const {
        return uninitialized() ? Tensor() : Tensor(placement, TensorDescriptor(newDataType, descriptor.getDimensions()));
    }
    Tensor clone(TensorPlacement newPlacement, DataType newDataType) const {
        return uninitialized() ? Tensor() : Tensor(newPlacement, TensorDescriptor(newDataType, descriptor.getDimensions()));
    }
    Tensor clone(std::vector<uint64_t> newDimensions) const {
        return uninitialized() ? Tensor() : Tensor(placement, TensorDescriptor(getDataType(), newDimensions));
    }

    TensorPlacement getPlacement() const { return placement; }
    template <typename ElementDataType = void>
    ElementDataType *getMemPtr();
    template <typename ElementDataType = void>
    const ElementDataType *getMemPtr() const;
    template <typename ElementDataType>
    ElementDataType getElement(std::vector<uint64_t> dimensionIndex);
    template <typename ElementDataType>
    void setElement(std::vector<uint64_t> dimensionIndex, const ElementDataType &value);
    template <typename ElementDataType = void>
    ElementDataType *getElementPointer(std::vector<uint64_t> dimensionIndex);
    TensorDescriptor getDescriptor() const;

    uint64_t getTensorId() const { return isInitialized() ? instanceId : 0; }

    void copyFromAsync(Tensor source, Stream stream);

    void downloadSection(Tensor &source, Stream &stream, uint64_t sourceOffset, uint64_t destOffset, uint64_t sizeBytes);
    void uploadSection(Tensor &dest, Stream &stream, uint64_t sourceOffset, uint64_t destOffset, uint64_t sizeBytes);

    // The values are set at the end of stream
    static Tensor zeros(TensorPlacement placement, TensorDescriptor descriptor, Stream stream);
    static Tensor randoms(TensorPlacement placement, TensorDescriptor descriptor, Stream stream, double minValue, double maxValue);
    static Tensor values(TensorPlacement placement, TensorDescriptor descriptor, Stream stream, double value);
    static Tensor identityMatrix(uint32_t N, TensorPlacement placement, DataType dataType, Stream stream);

    // numElements = 0 indicates all elements
    // Note that this takes num elements as its parameter rather than num bytes like regular memset
    // however the memory is set per byte like other versions of memset. To make this clear, value is int8_t.
    void memset(int8_t value, uint64_t numElements = 0);
    void memsetAsync(Stream stream, int8_t value, uint64_t numElements = 0);

    // Convert this tensor to refer to an uninitialized tensor
    // If this is the only reference to this tensor, its resources (memory) will be freed
    // Freeing of resources happens immediately, so you must ensure that there are no pending
    // accesses of the tensor's memory still enqueued on a stream, when the reference is dropped
    // FIXME, TODO: I should handle this internally by keeping track of pending operations and host side synchronizing
    //        with them before destroying the tensor. Consider if a tensor is allocated in a function, used to copy
    //        data onto the GPU and then the function returns. There could be a good amount of work already enqueued
    //        on the stream so it will be a while before the tensor memory is accessed, however the tensor is freed
    //        when the function returns since it had the only reference.
    //        Could I somehow make a temporary reference, perhaps in a future, where that process will wait till the operation
    //        completes and then drops the reference.
    //
    //        This is the fix: push (tensor, event) onto a static WorkQueue. Upon popping work queue synchronizes on the event and then
    //        returns So there is no concern about calling drop reference, or just references going out of scope, when there is future work
    //        associated with a tensor. But... am I detecting all work, copies are once thing, but what if its memory is being used in a
    //        stream? func() -> C = A + B; return C; Ok, I just need to ensure this happens on all tensor operations. Also I need unbounded
    //        no output work queue: unbounded loose end queue, uses a queue<pair<Tensor, Event>>.
    //
    // Warning! Ensure that all async work involving this tensor has been synchronized on the host before calling dropReference()!
    //
    // A correct pattern:
    // Tensor tensorFp32(cpuPlacement, descriptorFp32);
    // Tensor tensorFp16(cpuPlacement, descriptorFp16);
    // tensorFp16.copyFromAsync(tensorFp32, stream);
    // stream.synchronize();    <---- this is needed
    // tensorFp32.dropReference();
    void dropReference() { *this = Tensor(); }

    // Note minValue and maxValue are igorned for boolean types.
    void fillRandom(double minValue, double maxValue, Stream stream);
    void fillZero(Stream dataStream);

    void reshape(std::vector<uint64_t> dimensions);
    [[nodiscard]] Tensor aliasView(std::vector<uint64_t> dimensions,
                                   std::vector<uint64_t> strides_elements,
                                   uint64_t element_offset = 0) const;
    [[nodiscard]] bool hasCustomStrides() const { return !customStridesElements.empty(); }
    [[nodiscard]] bool isDenseContiguous() const;
    [[nodiscard]] uint64_t getStorageElementOffset() const { return storageElementOffset; }
    [[nodiscard]] std::vector<uint64_t> getStridesElements() const;
    // void concatenateFrom(std::vector<Tensor> sources);
    // void splitInto(std::vector<Tensor> destinations);

    void fill(const double value, Stream stream);

    // The scalar is cast to the type of the argument tensor, same behavior for the other scalar operations:
    // These functions perform the operation on the source tensor and write into this tensor
    // Both tensors must be on the same device.

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;
    bool operator<(const Tensor &other) const;

    // Convenience functions to pass information from the descriptor
    DataType getDataType() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return descriptor.getDataType();
    }
    std::vector<uint64_t> getDimensions() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return descriptor.getDimensions();
    }
    uint32_t getNumDimensions() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return descriptor.getNumDimensions();
    }
    uint64_t getTotalNumElements() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return descriptor.getTotalNumElements();
    }
    uint64_t getArraySizeInBytes() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return descriptor.getArraySizeInBytes();
    }

    std::string dimensionsToString();

    static uint32_t getThreadIdHash(uint32_t seed = 0);
    static uint64_t getThreadIdHash64(uint64_t seed = 0);

   private:
    void copyFromAsyncImpl(Tensor source, Stream copyStream);

    // BackingMemory may carry host-side caches derived from its payload. Any
    // generic mutation invalidates those caches; the structural owner (currently
    // RowPartitionRuntime) republishes them only after the new payload is known.
    //
    // This metadata follows the same single-owner host scheduling contract as
    // mutable tensor payload. It is not a cross-thread synchronization primitive:
    // producer/consumer transitions must already occur through Thor's synchronized
    // queue/session handoff before another host thread reads or mutates it.
    void invalidatePayloadDerivedRuntimeMetadata() {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostActiveValueCount.reset();
        backingMemory->rowPartitionHostMaxActiveRowLength.reset();
        backingMemory->rowPartitionHostOffsets.reset();
    }

    void setRowPartitionHostActiveValueCount(uint64_t activeValueCount) {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostActiveValueCount = activeValueCount;
    }
    void clearRowPartitionHostActiveValueCount() {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostActiveValueCount.reset();
    }
    [[nodiscard]] std::optional<uint64_t> getRowPartitionHostActiveValueCount() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return backingMemory->rowPartitionHostActiveValueCount;
    }
    void setRowPartitionHostMaxActiveRowLength(uint64_t maxActiveRowLength) {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostMaxActiveRowLength = maxActiveRowLength;
    }
    void clearRowPartitionHostMaxActiveRowLength() {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostMaxActiveRowLength.reset();
    }
    [[nodiscard]] std::optional<uint64_t> getRowPartitionHostMaxActiveRowLength() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return backingMemory->rowPartitionHostMaxActiveRowLength;
    }
    void setRowPartitionHostOffsets(std::vector<uint64_t> hostOffsets) {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostOffsets = std::move(hostOffsets);
    }
    void clearRowPartitionHostOffsets() {
        THOR_THROW_IF_FALSE(!uninitialized());
        backingMemory->rowPartitionHostOffsets.reset();
    }
    [[nodiscard]] std::optional<std::vector<uint64_t>> getRowPartitionHostOffsets() const {
        THOR_THROW_IF_FALSE(!uninitialized());
        return backingMemory->rowPartitionHostOffsets;
    }

    TensorPlacement placement;
    struct BackingMemory {
        explicit BackingMemory(TensorPlacement placement) : placement(placement) {}
        ~BackingMemory() noexcept;

        void releaseChecked();

        TensorPlacement placement;
        void *mem = nullptr;
        bool cpuMemPinnedViaCudaHostRegister = false;
        std::optional<uint64_t> rowPartitionHostActiveValueCount;
        std::optional<uint64_t> rowPartitionHostMaxActiveRowLength;
        std::optional<std::vector<uint64_t>> rowPartitionHostOffsets;
    };

    std::shared_ptr<BackingMemory> backingMemory;
    uint64_t storageElementOffset = 0;
    uint64_t storageNumElements = 0;
    std::vector<uint64_t> customStridesElements;

    uint64_t instanceId = 0;

    TensorDescriptor descriptor;

    // FIXME: get rid of this override descriptor nonsense
    bool descriptorOverridden = false;
    TensorDescriptor overriddenDescriptor;

    static std::atomic<uint64_t> nextInstanceId;

    void *getBaseMemPtr() const;
    void allocateMemory(uint32_t alignmentBytes = 0);
    bool uninitialized() const { return backingMemory == nullptr; }

    template <typename T>
    void launchFillValueGpuKernel(T value, T *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
    void fillGpuIdentityMatrixOnes(Stream stream);
    template <typename DATA_TYPE>
    void launchGpuFillRandom(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);

    void overrideDescriptor(TensorDescriptor overrideDescriptor);
    void clearDescriptorOverride();

    void construct(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes);
};

}  // namespace ThorImplementation
