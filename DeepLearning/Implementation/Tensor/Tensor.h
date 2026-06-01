#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"
#include "Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h"
#include "Utilities/WorkQueue/WorkQueueUnordered.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
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

/**
 * A multidimensional array that is allocated in either cpu or device mem.
 *
 * Note: copyFromAsync preserves the source value whereas moveFromAsync does not necessarily
 *       preserve the source value. The only thing that move buys you is that it allows
 *       copying across devices where the destination data type is smaller than the source data type.
 */

class Tensor : private ReferenceCounted {
   public:
    Tensor();
    Tensor(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes = 0);
    Tensor(const Tensor &tensorInstance);
    Tensor &operator=(const Tensor &tensorInstance);
    virtual ~Tensor();

    bool isInitialized() const { return !uninitialized(); }

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

    uint64_t getTensorId() const { return instanceId; }

    void copyFromAsync(Tensor source, Stream stream);

    void moveFromAsync(Tensor source, Stream stream);

    void downloadSection(Tensor &source, Stream &stream, uint64_t sourceOffset, uint64_t destOffset, uint64_t sizeBytes);
    void uploadSection(Tensor &dest, Stream &stream, uint64_t sourceOffset, uint64_t destOffset, uint64_t sizeBytes);

    enum class FileAccess { INVALID = 0, READ_ONLY, WRITE_ONLY, READ_WRITE };
    void attachFile(const std::string &fileName,
                    const off_t fileOffset,
                    const FileAccess fileAccessRequirement,
                    bool createEmptyFile = false);
    void attachFile(const std::string &fileName, const off_t fileOffset, const FileAccess fileAccessRequirement, int32_t fileDescriptor);
    void detachFile();
    std::string getAttachedFilename() { return fileName; }
    void loadFromFile(Stream stream, std::optional<uint32_t> crc = std::nullopt);
    void dumpToFile(Stream stream);

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
    void resize(std::vector<uint64_t> dimensions);
    void swapBackingMemoryWith(Tensor &other);
    // void concatenateFrom(std::vector<Tensor> sources);
    // void splitInto(std::vector<Tensor> destinations);

    void fill(const double value, Stream stream);

    // The scalar is cast to the type of the argument tensor, same behavior for the other scalar operations:
    // These functions perform the operation on the source tensor and write into this tensor
    // Both tensors must be on the same device.

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;
    bool operator<(const Tensor &other) const;

    using ReferenceCounted::getReferenceCount;

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
    void copyFromAsync(Tensor source, Stream copyStream, bool mustPreserveSourceValue);

    TensorPlacement placement;
    struct BackingMemory {
        void *mem = nullptr;
        bool cpuMemPinnedViaCudaHostRegister = false;
    };

    std::shared_ptr<BackingMemory> backingMemory;
    uint64_t storageElementOffset = 0;
    uint64_t storageNumElements = 0;
    std::vector<uint64_t> customStridesElements;

    uint64_t instanceId;

    TensorDescriptor descriptor;

    std::string fileName;
    int32_t fileDescriptor = 0;
    bool ownsFileDescriptor = false;
    FileAccess fileAccessRequirement;
    off_t fileOffset;

    // FIXME: get rid of this override descriptor nonsense
    bool descriptorOverridden = false;
    TensorDescriptor overriddenDescriptor;

    static std::atomic<uint64_t> nextInstanceId;

    void *getBaseMemPtr() const;
    void allocateMemory(uint32_t alignmentBytes = 0);

    template <typename T>
    void launchFillValueGpuKernel(T value, T *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
    void fillGpuIdentityMatrixOnes(Stream stream);
    template <typename DATA_TYPE>
    void launchGpuFillRandom(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);

    void overrideDescriptor(TensorDescriptor overrideDescriptor);
    void clearDescriptorOverride();

    void construct(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes);
    void copyObject(const Tensor &other);
    void destroy();
};

}  // namespace ThorImplementation
