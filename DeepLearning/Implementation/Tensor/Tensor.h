#pragma once

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/TypeConversions/TypeConverter.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <assert.h>

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

using std::deque;
using std::pair;
using std::set;
using std::unordered_map;
using std::vector;

using std::atomic;
using std::mutex;
using std::recursive_mutex;
using std::unique_lock;

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
    Tensor(TensorPlacement placement, TensorDescriptor descriptor);
    Tensor(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory);
    Tensor(const Tensor &tensorInstance);
    Tensor &operator=(const Tensor &tensorInstance);
    virtual ~Tensor();

    bool isInitialized() { return !uninitialized(); }
    bool isUsingExternallyManagedMemory();

    Tensor clone() { return uninitialized() ? Tensor() : Tensor(placement, descriptor); }
    Tensor clone(TensorPlacement newPlacement) { return uninitialized() ? Tensor() : Tensor(newPlacement, descriptor); }
    Tensor clone(TensorDescriptor::DataType newDataType) {
        return uninitialized() ? Tensor() : Tensor(placement, TensorDescriptor(newDataType, descriptor.getDimensions()));
    }
    Tensor clone(TensorPlacement newPlacement, TensorDescriptor::DataType newDataType) {
        return uninitialized() ? Tensor() : Tensor(newPlacement, TensorDescriptor(newDataType, descriptor.getDimensions()));
    }
    Tensor clone(vector<uint64_t> newDimensions) {
        return uninitialized() ? Tensor() : Tensor(placement, TensorDescriptor(getDataType(), newDimensions));
    }

    TensorPlacement getPlacement() { return placement; }
    void *getMemPtr() { return mem; }
    void *getElement(vector<unsigned long> dimensionIndex);
    TensorDescriptor getDescriptor();
    unsigned long getTensorId() { return instanceId; }

    void copyFromAsync(Tensor source, Stream stream);

    void moveFromAsync(Tensor source, Stream stream);

    void reshape(vector<unsigned long> dimensions);
    void resize(vector<unsigned long> dimensions);
    void concatenateFrom(vector<Tensor> sources);
    void splitInto(vector<Tensor> destinations);

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;
    bool operator<(const Tensor &other) const;

    using ReferenceCounted::getReferenceCount;

    // Convenience functions to pass information from the descriptor
    TensorDescriptor::DataType getDataType() { return descriptor.getDataType(); }
    vector<unsigned long> getDimensions() { return descriptor.getDimensions(); }
    unsigned int getNumDimensions() { return descriptor.getNumDimensions(); }
    unsigned long getTotalNumElements() { return descriptor.getTotalNumElements(); }
    long unsigned getArraySizeInBytes() { return descriptor.getArraySizeInBytes(); }

    std::string dimensionsToString();

   private:
    void copyFromAsync(Tensor source, Stream copyStream, bool mustPreserveSourceValue);

    TensorPlacement placement;
    void *mem;

    unsigned long instanceId;

    TensorDescriptor descriptor;

    bool usingExternallyManagedMemory;

    // FIXME: get rid of this override descriptor nonsense
    bool descriptorOverridden;
    TensorDescriptor overriddenDescriptor;

    void allocate();
    void deallocate();

    static atomic<unsigned long> nextInstanceId;

    void allocateMemory();

    void overrideDescriptor(TensorDescriptor descriptor);
    void clearDescriptorOverride();

    void construct(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory);
    void copyObject(const Tensor &other);
    void destroy();
};

}  // namespace ThorImplementation
