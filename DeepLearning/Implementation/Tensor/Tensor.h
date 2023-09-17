#pragma once

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/ReferenceCounted.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/GpuMatrixTranspose/gpuMatrixTranspose.h"
#include "Utilities/TensorOperations/TypeConversions/TypeConverter.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <assert.h>

#include <cuda.h>
#include <cuda_fp16.h>
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
    template <typename ElementDataType = void>
    ElementDataType *getMemPtr();
    void *getElement(std::vector<unsigned long> dimensionIndex);
    TensorDescriptor getDescriptor();

    unsigned long getTensorId() { return instanceId; }

    void copyFromAsync(Tensor source, Stream stream);

    void moveFromAsync(Tensor source, Stream stream);

    // numElements = 0 indicates all elements
    // Note that this takes num elements as its parameter rather than num bytes like regular memset
    // however the memory is set per byte like other versions of memset. To make this clear, value is int8_t.
    void memset(int8_t value, uint64_t numElements = 0);
    void memsetAsync(Stream stream, int8_t value, uint64_t numElements = 0);
    void clear();
    void clearAsync(Stream stream);

    // setValues is intended as a test helper to easily populate an entire tensor
    // It is less efficent than working with tensor memory directly since it uses non-pinned cpu memory and is not meant to be used
    // in performance critical code.
    template <typename T>
    void setValues(std::vector<T> values, Stream stream);
    // loadValuesIntoVector is also only a test helper and should not be used in production code.
    template <typename T>
    void loadValuesIntoVector(std::vector<T> &values, Stream stream);

    void reshape(std::vector<unsigned long> dimensions);
    void resize(std::vector<unsigned long> dimensions);
    void concatenateFrom(std::vector<Tensor> sources);
    void splitInto(std::vector<Tensor> destinations);

    template <typename T>
    void fill(const T value, Stream stream);

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
     * [thisTensor] = alpha * [augend] + addend, elementwise
     * <div/>
     * addend is first cast to the data type of augend.
     */
    void add(Tensor augend, double addend, float alpha, Stream stream);

    /**
     * [thisTensor] = augend + beta * [addend], elementwise
     * <div/>
     * augend is first cast to the data type of addend.
     */
    void add(double augend, Tensor addend, float beta, Stream stream);

    /**
     * [thisTensor] = alpha * [augend] + beta * [addend], elementwise
     * <div/>
     * augend and addend need to be of the same data type.
     */
    void add(Tensor augend, Tensor addend, float alpha, float beta, Stream stream);

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
     * [thisTensor] = the integer componet of [argument], elementwise
     * <div/>
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     */
    void truncateFloatingPoint(Tensor argument, Stream stream);

    /**
     * [thisTensor] = [a] * [b] + [c], elementwise
     * <div/>
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
     * [thisTensor] = erf([x]), elementwise
     * <div/>
     * Compute the error function: https://mathworld.wolfram.com/Erf.html
     * x must be float.
     * The return type may be float or half
     */
    void erf(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfinv([x]), elementwise
     * <div/>
     * Compute the inverse error function defined as erfinv(erf(x))=x : https://www.mathworks.com/help/symbolic/erfinv.html
     * x must be float.
     * The return type may be float or half
     */
    void erfinv(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfc([x]), elementwise
     * <div/>
     * Compute the complementary error function: https://mathworld.wolfram.com/Erfc.html
     * x must be float.
     * The return type may be float or half
     */
    void erfc(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfcinv([x]), elementwise
     * <div/>
     * Compute the inverse complementary error function defined as erfcinv(erfc(x))=x :
     * https://www.mathworks.com/help/matlab/ref/erfcinv.html#bup512o-2 x must be float. The return type may be float or half
     */
    void erfcinv(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfcx([x]), elementwise
     * <div/>
     * Compute the scaled complementary error function that is equal to exp(x^2)*erfc(x):
     * https://www.mathworks.com/help/matlab/ref/erfcx.html x must be float. The return type may be float or half
     */
    void erfcx(Tensor x, Stream stream);

    /**
     * [thisTensor] = gamma([x]), elementwise
     * <div/>
     * Compute the gamma(x): https://mathworld.wolfram.com/GammaFunction.html
     * x must be float. The return type may be float or half
     */
    void tgamma(Tensor x, Stream stream);

    /**
     * [thisTensor] = ln(gamma([x])), elementwise
     * <div/>
     * gamma(x): https://mathworld.wolfram.com/GammaFunction.html
     * x must be float. The return type may be float or half
     */
    void lgamma(Tensor x, Stream stream);

    /**
     * [thisTensor] = sin([radians]), elementwise
     * <div/>
     * Compute the sine of radians
     * r must be float. The return type may be float or half
     */
    void sin(Tensor radians, Stream stream);

    /**
     * [thisTensor] = cos([radians]), elementwise
     * <div/>
     * Compute the cosine of radians
     * r must be float. The return type may be float or half
     */
    void cos(Tensor radians, Stream stream);

    /**
     * [thisTensor] = tan([radians]), elementwise
     * <div/>
     * Compute the tangent of radians
     * r must be float. The return type may be float or half
     */
    void tan(Tensor radians, Stream stream);

    /**
     * [thisTensor] = csc([radians]), elementwise
     * <div/>
     * Compute the cosecant of radians
     * r must be float. The return type may be float or half
     */
    void csc(Tensor radians, Stream stream);

    /**
     * [thisTensor] = sec([radians]), elementwise
     * <div/>
     * Compute the secant of radians
     * r must be float. The return type may be float or half
     */
    void sec(Tensor radians, Stream stream);

    /**
     * [thisTensor] = cot([radians]), elementwise
     * <div/>
     * Compute the cotangent of radians
     * r must be float. The return type may be float or half
     */
    void cot(Tensor radians, Stream stream);

    /**
     * [thisTensor] = asin([radians]), elementwise
     * <div/>
     * Compute the arcsine of radians
     * r must be float. The return type may be float or half
     */
    void asin(Tensor radians, Stream stream);

    /**
     * [thisTensor] = acos([radians]), elementwise
     * <div/>
     * Compute the arccosine of radians
     * r must be float. The return type may be float or half
     */
    void acos(Tensor radians, Stream stream);

    /**
     * [thisTensor] = atan([radians]), elementwise
     * <div/>
     * Compute the arctangent of radians
     * r must be float. The return type may be float or half
     */
    void atan(Tensor radians, Stream stream);

    /**
     * [thisTensor] = acsc([radians]), elementwise
     * <div/>
     * Compute the arccosecant of radians
     * r must be float. The return type may be float or half
     */
    void acsc(Tensor radians, Stream stream);

    /**
     * [thisTensor] = asec([radians]), elementwise
     * <div/>
     * Compute the arcsecant of radians
     * r must be float. The return type may be float or half
     */
    void asec(Tensor radians, Stream stream);

    /**
     * [thisTensor] = acot([radians]), elementwise
     * <div/>
     * Compute the arccotangent of radians
     * r must be float. The return type may be float or half
     */
    void acot(Tensor radians, Stream stream);

    /**
     * [thisTensor] = sinh([radians]), elementwise
     * <div/>
     * Compute the hyperbolic sine of radians
     * r must be float. The return type may be float or half
     */
    void sinh(Tensor radians, Stream stream);

    /**
     * [thisTensor] = cosh([radians]), elementwise
     * <div/>
     * Compute the hyperbolic cosine of radians
     * r must be float. The return type may be float or half
     */
    void cosh(Tensor radians, Stream stream);

    /**
     * [thisTensor] = tanh([radians]), elementwise
     * <div/>
     * Compute the hyperbolic tangent of radians
     * r must be float. The return type may be float or half
     */
    void tanh(Tensor radians, Stream stream);

    /**
     * [thisTensor] = csch([radians]), elementwise
     * <div/>
     * Compute the hyperbolic cosecant of radians
     * r must be float. The return type may be float or half
     */
    void csch(Tensor radians, Stream stream);

    /**
     * [thisTensor] = sech([radians]), elementwise
     * <div/>
     * Compute the hyperbolic secant of radians
     * r must be float. The return type may be float or half
     */
    void sech(Tensor radians, Stream stream);

    /**
     * [thisTensor] = coth([radians]), elementwise
     * <div/>
     * Compute the hyperbolic cotangent of radians
     * r must be float. The return type may be float or half
     */
    void coth(Tensor radians, Stream stream);

    /**
     * [thisTensor] = asinh([radians]), elementwise
     * <div/>
     * Compute the hyperbolic arcsine of radians
     * r must be float. The return type may be float or half
     */
    void asinh(Tensor radians, Stream stream);

    /**
     * [thisTensor] = acosh([radians]), elementwise
     * <div/>
     * Compute the hyperbolic arccosine of radians
     * r must be float. The return type may be float or half
     */
    void acosh(Tensor radians, Stream stream);

    /**
     * [thisTensor] = atanh([radians]), elementwise
     * <div/>
     * Compute the hyperbolic arctangent of radians
     * r must be float. The return type may be float or half
     */
    void atanh(Tensor radians, Stream stream);

    /**
     * [thisTensor] = acsch([radians]), elementwise
     * <div/>
     * Compute the hyperbolic arccosecant of radians
     * r must be float. The return type may be float or half
     */
    void acsch(Tensor radians, Stream stream);

    /**
     * [thisTensor] = asech([radians]), elementwise
     * <div/>
     * Compute the hyperbolic arcsecant of radians
     * r must be float. The return type may be float or half
     */
    void asech(Tensor radians, Stream stream);

    /**
     * [thisTensor] = acoth([radians]), elementwise
     * <div/>
     * Compute the hyperbolic arccotangent of radians
     * r must be float. The return type may be float or half
     */
    void acoth(Tensor radians, Stream stream);

    /**
     * [thisTensor] = max(thisTensor[i], minValue)
     * <div/>
     * minValue is first cast to the data type of this tensor.
     */
    void max(double minValue, Stream stream);

    /**
     * [thisTensor] = max(thisTensor[i], minValues[i])
     */
    void max(Tensor minValues, Stream stream);

    /**
     * [thisTensor] = min(thisTensor[i], maxValue)
     * <div/>
     * minValue is first cast to the data type of this tensor.
     */
    void min(double maxValue, Stream stream);

    /**
     * [thisTensor] = min(thisTensor[i], maxValues[i])
     */
    void min(Tensor maxValues, Stream stream);

    /**
     * [thisTensor] = max(min(thisTensor[i], maxValue), minValue)
     * <div/>
     * minValue is first cast to the data type of this tensor.
     */
    void bound(double minValue, double maxValue, Stream stream);

    /**
     * returns a new tensor that equals [thisTensor]T
     *
     * FIXME: Currently only supported for tensors of type FP16 and FP32
     */
    Tensor transposeMatrix(Stream stream);

    /**
     * Transposes [thisTensor] in-place. Only valid for square tensors.
     *
     * FIXME: Currently only supported for tensors of type FP16 and FP32
     */
    void transposeSquareMatrixInPlace(Stream stream);

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;
    bool operator<(const Tensor &other) const;

    using ReferenceCounted::getReferenceCount;

    // Convenience functions to pass information from the descriptor
    TensorDescriptor::DataType getDataType() {
        assert(!uninitialized());
        return descriptor.getDataType();
    }
    std::vector<unsigned long> getDimensions() {
        assert(!uninitialized());
        return descriptor.getDimensions();
    }
    unsigned int getNumDimensions() {
        assert(!uninitialized());
        return descriptor.getNumDimensions();
    }
    unsigned long getTotalNumElements() {
        assert(!uninitialized());
        return descriptor.getTotalNumElements();
    }
    long unsigned getArraySizeInBytes() {
        assert(!uninitialized());
        return descriptor.getArraySizeInBytes();
    }

    std::string dimensionsToString();

    virtual bool isKerasCompatible(std::string &explanation) {
        explanation.clear();
        return true;
    }

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

    template <typename T>
    void launchFillValueGpuKernel(T value, T *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);

    void overrideDescriptor(TensorDescriptor descriptor);
    void clearDescriptorOverride();

    void construct(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory);
    void copyObject(const Tensor &other);
    void destroy();
};

}  // namespace ThorImplementation
