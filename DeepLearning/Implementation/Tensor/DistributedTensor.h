#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

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

class Tensor;

/**
 * A multidimensional array that is used to connect layers that learn via back propagation.
 *
 * Note that copying a tensor object to another tensor object is a light-weight operation,
 * but copying the memory contained by one tensor to a different tensor is a heavy weight operation.
 * For example, say you have Tensors t0 and t1:
 * t1 = t0;   <- this is a lightweight operation, both tensor objects refer to the same underlying memory.
 * however:
 * t1.copyFromAsync(t0, stream);   <- this is a heavy weight operation that copies t0's data to t1's memory.
 *
 * FIXME: I think I should rename this PhysicalTensor, and the inner class just Instance - maybe tensor and
 * distributedTensor Then I would create DistributedTensor, which is used to connect up layers, or maybe Add all the
 * functionality without implementations to tensor, so user can connect up the network via metadata, and Tensor would
 * hold the implementations of those functions.
 */

// Thread safety:
//      multiple threads can add and remove instances at the same time if instances are not being used to move data,
//      multiple threads can use the instances data if no threads are adding or removing instances
//
//      if multiple threads will be adding/removing instances and moving data, the threads must coordinate by
//      explicitly calling lock() on the DistributedTensor before performing the operations and then explicitly unlock()
//      the DistributedTensor afterward.
class DistributedTensor : private ReferenceCounted {
   public:
    DistributedTensor();
    DistributedTensor(TensorDescriptor descriptor);
    DistributedTensor(const DistributedTensor &tensor);
    DistributedTensor &operator=(const DistributedTensor &tensor);

    virtual ~DistributedTensor();

    TensorDescriptor getDescriptor() {
        assert(!uninitialized());
        return descriptor;
    }
    unsigned long getDistributedTensorId() const {
        assert(!uninitialized());
        return distributedTensorId;
    }

    // Implement the tensor manipulation functions that are back propagable
    // operators: +, -, *, /   actually, maybe not operators, I would prefer scalarDivide, elementwiseDivide, in general
    // scalar* elementwise* etc. higher order math functions: matrix multiply, outer product, inner product. <- For
    // these ensure the tensor is a matrix and dimensionality is legal for the operation.

    // Any dest tensor instance is free to copy from source tensor instance, when using this function, all souce tensor
    // instances should hold the same data.
    void copyFromAsync(DistributedTensor source, Stream stream);
    // All dest tensor instances will copy from the specified source tensor instance
    void copyFromAsync(Tensor source, Stream stream);

    // FIXME: create static functions on this class that take a few tensors and perform the compound operation.
    //
    // Useful when you have multiple instances of a neural network:
    // void reduceInstancesIntoInstance(int destInstanceIndex) { assert(false); /* TODO: implement */ }
    // Useful when you want to reduce into fp32 from an fp16 tensor for example:
    // void reduceTensorIntoInstance(int destInstanceIndex, DistributedTensor source) {
    //    assert(false); /* TODO: implement */
    //}

    Tensor addInstance(TensorPlacement instancePlacement);
    bool hasInstance(unsigned long instanceId);
    bool hasInstance(TensorPlacement tensorPlacement);
    Tensor getInstance(unsigned long instanceId);
    Tensor getInstance(TensorPlacement tensorPlacement);
    Tensor getNearestInstance(Tensor other);
    Tensor getNearestInstance(TensorPlacement tensorPlacement);
    unordered_map<unsigned long, Tensor> getInstances();
    void removeInstance(unsigned long instanceId);
    void removeInstance(TensorPlacement placement);

    unsigned long getNumInstances() {
        assert(!uninitialized());
        return instances->size();
    }
    Tensor getAnyInstance();

    bool operator==(const DistributedTensor &other) const;
    bool operator!=(const DistributedTensor &other) const;

    using ReferenceCounted::getReferenceCount;

    void lock() {
        assert(!uninitialized());
        tensorMutex->lock();
    }
    void unlock() {
        assert(!uninitialized());
        tensorMutex->unlock();
    }

   private:
    TensorDescriptor descriptor;

    unordered_map<unsigned long, Tensor> *instances;  // key is instanceId
    recursive_mutex *tensorMutex;

    unsigned long distributedTensorId;

    static atomic<unsigned long> nextTensorId;

    void copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                           map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                           Stream stream);
    map<int, Event> copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                                      map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                      Event startEvent);
    map<int, Event> copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                                      map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                      map<int, Event> &populatedEventPerDevice);
    map<int, Event> copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                                      map<int, Tensor> &unpopulatedInstancePerDevice,
                                      map<int, Event> &populatedEventPerDevice);

    void peformOnGpuConversions(map<int, Tensor> &populatedInstancePerDevice,
                                map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                Stream stream,
                                map<int, Tensor> &convertedInstancePerDevice,
                                map<int, Event> &populatedEventPerDevice);

    void crossDeviceCopy(int copyToGpuNum,
                         int copyFromGpuNum,
                         map<int, Tensor> &populatedInstancePerDevice,
                         map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                         map<int, Event> &populatedEventPerDevice);
    void localDeviceCopy(int gpuNum,
                         map<int, Tensor> &populatedInstancePerDevice,
                         map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                         map<int, Event> &populatedEventPerDevice);

    void removeAllInstances(TensorPlacement placement) {
        assert(!uninitialized());
        instances->clear();
    }

    void construct(TensorDescriptor descriptor);
    void copyObject(const DistributedTensor &other);
    void destroy();
};
