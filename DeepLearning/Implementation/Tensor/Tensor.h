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

    /**
     * [thisTensor] = [base] ^ exponent, elementwise
     * <div/>
     * base must have data type FP32.
     */
    void pow(Tensor base, float exponent, Stream stream);

    /**
     * [thisTensor] = base ^ [exponent], elementwise
     * <div/>
     * exponent must have data type FP32.
     * there is no restriction on the data type of this destination tensor.
     */
    void pow(float base, Tensor exponent, Stream stream);

    /**
     * [thisTensor] = [base] ^ [exponent], elementwise
     * <div/>
     * exponent and base must have data type FP32.
     * there is no restriction on the data type of this destination tensor.
     */
    void pow(Tensor base, Tensor exponent, Stream stream);

    /**
     * [thisTensor] = [base] ^ exponent, elementwise
     * <div/>
     * the computation will be done in FP32,
     * there is no restriction on the data type of this destination tensor.
     */
    void exp(float exponent, Stream stream);

    /**
     * [thisTensor] = base ^ [exponent], elementwise
     * <div/>
     * exponent must have data type FP32,
     * there is no restriction on the data type of this destination tensor.
     */
    void exp(Tensor exponent, Stream stream);

    /**
     * [thisTensor] = ln([argument]), elementwise
     * <div/>
     * Compute the natural log of the argument tensor
     * argument must be float or half
     * there is no restriction on the data type of this destination tensor.
     */
    void log(Tensor argument, Stream stream);

    /**
     * [thisTensor] = ln([argument]), elementwise
     * <div/>
     * Compute the log with the specified base of the argument tensor
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void log(Tensor argument, float base, Stream stream);

    /**
     * [thisTensor] = ⌈ [argument] ⌉, elementwise
     * <div/>
     * Compute the ceil of each element in the argument tensor
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void ceil(Tensor argument, Stream stream);

    /**
     * [thisTensor] = ⌊ [argument] ⌋, elementwise
     * <div/>
     * Compute the floor of each element in the argument tensor
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void floor(Tensor argument, Stream stream);

    /**
     * [thisTensor] = round([argument]), elementwise
     * <div/>
     * Round to nearest integer, 0.5 rounds up.
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void round(Tensor argument, Stream stream);

    /**
     * [thisTensor] = [a] * [b] + [c], elementwise
     * <div/>
     * Round to nearest integer, 0.5 rounds up.
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void multiplyAccumulate(Tensor a, Tensor b, Tensor c, Stream stream);

    /**
     * [thisTensor] = 1 / [argument], elementwise
     * <div/>
     * Compute the reciprocal of each element in the argument tensor
     * argument must be half. use divide for other data types.
     * there is no restriction on the data type of this destination tensor.
     */
    void reciprocal(Tensor argument, Stream stream);

    /**
     * [thisTensor] = √([argument]), elementwise
     * <div/>
     * Compute the square root of each element in the argument tensor
     * argument must be float or half.
     * base will be converted into the type of argument.
     * there is no restriction on the data type of this destination tensor.
     */
    void sqrt(Tensor argument, Stream stream);

    /**
     * [thisTensor] = 1 / sqrt([argument]), elementwise
     * <div/>
     * Compute the reciprocal of the square root of each element in the argument tensor
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void reciprocalSqrt(Tensor argument, Stream stream);

    /**
     * [thisTensor] = erf(x), elementwise
     * <div/>
     * Compute the error function: https://mathworld.wolfram.com/Erf.html
     * x must be float.
     * The return type may be float or half
     */
    void erf(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfinv(x), elementwise
     * <div/>
     * Compute the inverse error function defined as erfinv(erf(x))=x : https://www.mathworks.com/help/symbolic/erfinv.html
     * x must be float.
     * The return type may be float or half
     */
    void erfinv(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfc(x), elementwise
     * <div/>
     * Compute the complementary error function: https://mathworld.wolfram.com/Erfc.html
     * x must be float.
     * The return type may be float or half
     */
    void erfc(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfcinv(x), elementwise
     * <div/>
     * Compute the inverse complementary error function defined as erfcinv(erfc(x))=x :
     * https://www.mathworks.com/help/matlab/ref/erfcinv.html#bup512o-2 x must be float. The return type may be float or half
     */
    void erfcinv(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfcx(x), elementwise
     * <div/>
     * Compute the scaled complementary error function that is equal to exp(x^2)*erfc(x):
     * https://www.mathworks.com/help/matlab/ref/erfcx.html x must be float. The return type may be float or half
     */
    void erfcx(Tensor x, Stream stream);

    // FIXME: expand this pattern to cover all useful functions...
    // abs
    // sin(), cos(), trig functions

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
