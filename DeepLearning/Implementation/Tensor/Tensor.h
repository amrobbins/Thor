#pragma once

#include "TensorDescriptor.h"
#include "TensorPlacement.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/TypeConversions/TypeConverter.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <set>
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

class DistributedTensor;

/**
 * A multidimensional array that is allocated in either cpu or device mem.
 *
 * Note: copyFromAsync preserves the source value whereas moveFromAsync does not necessarily
 *       preserve the source value. The only thing that move buys you is that it allows
 *       copying across devices where the destination data type is smaller than the source data type.
 */

class Tensor {
   public:
    Tensor();
    Tensor(TensorPlacement placement, TensorDescriptor descriptor);
    Tensor(const Tensor &tensorInstance);
    Tensor &operator=(const Tensor &tensorInstance);
    virtual ~Tensor();

    bool isInitialized() { return !uninitialized; }

    Tensor clone() { return uninitialized ? Tensor() : Tensor(placement, descriptor); }
    Tensor clone(TensorPlacement newPlacement) { return uninitialized ? Tensor() : Tensor(newPlacement, descriptor); }
    Tensor clone(TensorDescriptor::DataType newDataType) {
        return uninitialized ? Tensor() : Tensor(placement, TensorDescriptor(newDataType, descriptor.getDimensions()));
    }

    TensorPlacement getPlacement() { return placement; }
    void *getMemPtr() { return mem; }
    void *getElement(vector<unsigned long> dimensionIndex);
    TensorDescriptor getDescriptor();
    unsigned long getTensorId() { return instanceId; }

    void copyFromAsync(Tensor source, Stream stream);
    void copyFromAsync(DistributedTensor source, Stream stream);

    void moveFromAsync(Tensor source, Stream stream);
    void moveFromAsync(DistributedTensor source, Stream stream);

    // The following function variants return an event that indicates that the copying is finished when
    // cudaStreamWaitEvent() is called on this event.
    Event copyFromAsync(Tensor source, Event startEvent);
    Event copyFromAsync(DistributedTensor source, Event startEvent);

    Event moveFromAsync(Tensor source, Event startEvent);
    Event moveFromAsync(DistributedTensor source, Event startEvent);

    void reshape(vector<unsigned long> dimensions);
    void concatenateFrom(vector<Tensor> sources);
    void splitInto(vector<Tensor> destinations);

    int getReferenceCount() { return *referenceCount; }

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;

   private:
    void copyFromAsync(Tensor source, Stream stream, bool mustPreserveSourceValue);
    void copyFromAsync(DistributedTensor source, Stream stream, bool mustPreserveSourceValue);
    Event copyFromAsync(Tensor source, Event startEvent, bool mustPreserveSourceValue);
    Event copyFromAsync(DistributedTensor source, Event startEvent, bool mustPreserveSourceValue);

    bool uninitialized;

    TensorPlacement placement;
    void *mem;

    atomic<int> *referenceCount;

    unsigned long instanceId;

    TensorDescriptor descriptor;

    // FIXME: get rid of this override descriptor nonsense
    bool descriptorOverridden;
    TensorDescriptor overriddenDescriptor;

    void allocate();
    void deallocate();

    static atomic<unsigned long> nextInstanceId;

    void overrideDescriptor(TensorDescriptor descriptor);
    void clearDescriptorOverride();

    friend class DistributedTensor;
};
