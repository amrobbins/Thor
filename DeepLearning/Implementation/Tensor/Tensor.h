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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
    Tensor clone(std::vector<uint64_t> newDimensions) {
        return uninitialized() ? Tensor() : Tensor(placement, TensorDescriptor(getDataType(), newDimensions));
    }

    TensorPlacement getPlacement() { return placement; }
    void *getMemPtr() { return mem; }
    void *getElement(std::vector<unsigned long> dimensionIndex);
    TensorDescriptor getDescriptor();

    unsigned long getTensorId() { return instanceId; }

    void copyFromAsync(Tensor source, Stream stream);

    void moveFromAsync(Tensor source, Stream stream);

    void reshape(std::vector<unsigned long> dimensions);
    void resize(std::vector<unsigned long> dimensions);
    void concatenateFrom(std::vector<Tensor> sources);
    void splitInto(std::vector<Tensor> destinations);

    // The scalar is casted to the type of the argument tensor, same behavior for the other scalar operations:
    // These functions perform the operation on the source tensor and write into this tensor
    // Both tensors must be on the same device.

    /**
     * [thisTensor] = [augend] + addend, elementwise
     * <div/>
     * addend is first cast to the data type of augend.
     */
    void add(Tensor augend, double addend, Stream stream);

    /**
     * [thisTensor] = augend + [addend], elementwise
     * <div/>
     * augend is first cast to the data type of addend.
     */
    void add(double augend, Tensor addend, Stream stream);

    /**
     * [thisTensor] = [augend] + [addend], elementwise
     * <div/>
     * augend and addend need to be of the same data type.
     */
    void add(Tensor augend, Tensor addend, Stream stream);

    /**
     * [thisTensor] = [minuend] - subtrahend, elementwise
     * <div/>
     * subtrahend is first cast to the data type of minuend.
     */
    void subtract(Tensor minuend, double subtrahend, Stream stream);

    /**
     * [thisTensor] = minuend - [subtrahend], elementwise
     * <div/>
     * minuend is first cast to the data type of subtrahend.
     */
    void subtract(double minuend, Tensor subtrahend, Stream stream);

    /**
     * [thisTensor] = [minuend] - [subtrahend], elementwise
     * <div/>
     * minuend and subtrahend need to be of the same data type.
     */
    void subtract(Tensor minuend, Tensor subtrahend, Stream stream);

    /**
     * [thisTensor] = [multiplicand] * multiplier, elementwise
     * <div/>
     * multiplier is first cast to the data type of multiplicand.
     */
    void multiply(Tensor multiplicand, double multiplier, Stream stream);

    /**
     * [thisTensor] = multiplicand * [multiplier], elementwise
     * <div/>
     * multiplicand is first cast to the data type of multiplier.
     */
    void multiply(double multiplicand, Tensor multiplier, Stream stream);

    /**
     * [thisTensor] = [multiplicand] * [multiplier], elementwise
     * <div/>
     * multiplicand and multiplier need to be of the same data type.
     */
    void multiply(Tensor multiplicand, Tensor multiplier, Stream stream);

    /**
     * [thisTensor] = [numerator] / denominator, elementwise
     *
     * denominator is first cast to the data type of numerator.
     */
    void divide(Tensor numerator, double denominator, Stream stream);

    /**
     * [thisTensor] = numerator / [denominator], elementwise
     * <div/>
     * numerator is first cast to the data type of denominator.
     */
    void divide(double numerator, Tensor denominator, Stream stream);

    /**
     * [thisTensor] = [numerator] / [denominator], elementwise
     * <div/>
     * numerator and denominator need to be of the same data type.
     */
    void divide(Tensor numerator, Tensor denominator, Stream stream);

    // pow(float power)
    // log(float base)
    // ln()
    // abs()
    // trig functions

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;
    bool operator<(const Tensor &other) const;

    using ReferenceCounted::getReferenceCount;

    // Convenience functions to pass information from the descriptor
    TensorDescriptor::DataType getDataType() { return descriptor.getDataType(); }
    std::vector<unsigned long> getDimensions() { return descriptor.getDimensions(); }
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

    static std::atomic<unsigned long> nextInstanceId;

    void allocateMemory();

    void overrideDescriptor(TensorDescriptor descriptor);
    void clearDescriptorOverride();

    void construct(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory);
    void copyObject(const Tensor &other);
    void destroy();
};

}  // namespace ThorImplementation
