#pragma once

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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufile.h>
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
    Tensor(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory);
    Tensor(const Tensor &tensorInstance);
    Tensor &operator=(const Tensor &tensorInstance);
    virtual ~Tensor();

    bool isInitialized() { return !uninitialized(); }
    bool isUsingExternallyManagedMemory();

    Tensor clone() const { return uninitialized() ? Tensor() : Tensor(placement, descriptor); }
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

    TensorPlacement getPlacement() const { return placement; }
    template <typename ElementDataType = void>
    ElementDataType *getMemPtr();
    template <typename ElementDataType>
    ElementDataType getElement(std::vector<unsigned long> dimensionIndex);
    template <typename ElementDataType>
    void setElement(std::vector<unsigned long> dimensionIndex, const ElementDataType &value);
    template <typename ElementDataType = void>
    ElementDataType *getElementPointer(std::vector<unsigned long> dimensionIndex);
    TensorDescriptor getDescriptor() const;

    unsigned long getTensorId() { return instanceId; }

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
    void loadFromFile(Stream stream, Optional<uint32_t> crc = Optional<uint32_t>::empty());
    void dumpToFile(Stream stream);

    // The values are set at the end of stream
    static Tensor zeros(TensorPlacement placement, TensorDescriptor descriptor, Stream stream);
    static Tensor randoms(TensorPlacement placement, TensorDescriptor descriptor, Stream stream, double minValue, double maxValue);
    static Tensor values(TensorPlacement placement, TensorDescriptor descriptor, Stream stream, double value);
    static Tensor identityMatrix(uint32_t N, TensorPlacement placement, TensorDescriptor::DataType dataType, Stream stream);

    // numElements = 0 indicates all elements
    // Note that this takes num elements as its parameter rather than num bytes like regular memset
    // however the memory is set per byte like other versions of memset. To make this clear, value is int8_t.
    void memset(int8_t value, uint64_t numElements = 0);
    void memsetAsync(Stream stream, int8_t value, uint64_t numElements = 0);
    void clear();
    void clearAsync(Stream stream);

    // Convert this tensor to refer to an uninitialized tensor
    // If this is the only reference to this tensor, its resources (memory) will be freed
    // Freeing of resources happens immediately, so you must ensure that there are no pending
    // accesses of the tensor's memory still enqueued on a stream, when the reference is dropped
    // FIXME: I should handle this internally by keeping track of pending operations and host side synchronizing
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

    void reshape(std::vector<unsigned long> dimensions);
    void resize(std::vector<unsigned long> dimensions);
    void concatenateFrom(std::vector<Tensor> sources);
    void splitInto(std::vector<Tensor> destinations);

    void fill(const double value, Stream stream);

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
     * This operation is defined by the shape of the input tensors.
     *
     * All 1 dimensional tensors will be interpreted as having 2 dimensions with a size 1 second (columns) dimension.
     *
     * 1. If either input tensor is one element, then this will result in a tensor scaling operation. (i.e. scalar broadcast multiplication)
     * 2. If both inputs are vectors of the same shape, an element-wise multiplication will be performed.
     *    i.e. Two column vectors of dimensions (N,1) or two row vectors of dimensions (1,N)
     * 3. If both inputs are matrices of compatible sizes then a matrix multiplication will be performed.
     *    Note that there is no overlap between cases 2 and 3 except where both tensors contain a single element, in which case scalar
     *    multiplication will be performed as described in (1).
     * 4. If one tensor has more than 2 dimensions and the other tensor is not a scalar, the operation is not supported.
     *
     * </div>
     * Note: for case 3 (matrix matrix multiply) using CublasMatrixMultiply and finding the optimal kernel is preferred, whereas this
     * version uses a heuristic kernel choice. Also CublasMatrixMultiply gives more options, such as scaling and transposing and allowing
     * non-packed leading dimensions. CublasMatrixMultiply is used for the implementation here, where its full API is not exposed.
     *
     * <div/>
     * multiplicand and multiplier need to be of the same data type. And for case 3 that data type must be FP16 or FP32.
     */
    void multiply(Tensor multiplicand, Tensor multiplier, Stream stream);

    /**
     * [thisTensor] = alpha * [A] * [B] + beta * [C]
     *
     * The shape of all tensors must be matrices with dimensions that are valid for this computation, specifically that:
     * A_cols == B_rows, A_rows == C_rows, B_cols == C_cols, thisTensor_dimensions = C_dimensions.
     * Tensors with 1 dimension are taken to be a matrix with a column size of 1.
     * When beta = 0.0f, then C is not loaded and can be empty.
     *
     * </div>
     * Note: Using CublasMatrixMultiply and finding the optimal kernel is preferred, whereas this version uses a heuristic kernel choice.
     *       Also CublasMatrixMultiply gives more options, such as scaling and transposing and allowing non-packed leading dimensions.
     *       CublasMatrixMultiply is used for the implementation here, where its full API is not exposed.
     *
     * <div/>
     * A, B and C must be the same data type and must be either FP16 or FP32.
     */
    void gemm(Tensor A, Tensor B, Optional<Tensor> C, float alpha, float beta, Stream stream);

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
     * [thisTensor] = A · B
     * <div/>
     * AKA inner product. All tensors must be the same data type.
     */
    void dotProduct(Tensor A, Tensor B, Stream stream);

    /**
     * [thisTensor] = A ⊗ B
     * <div/>
     * All tensors must be the same data type.
     */
    void outerProduct(Tensor A, Tensor B, Stream stream);

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
     * [thisTensor] = e ^ [exponent], elementwise
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
     * [thisTensor] = log_base([argument]), elementwise
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
     * [thisTensor] = [tensor] * scalar, elementwise
     *
     * <div/>
     * Both tensors need to be of the same type.
     */
    void multiplyTensorScalar(Tensor tensor, Tensor scalar, Stream stream);

    /**
     * [thisTensor] = [tensor] * scalar, elementwise
     *
     * <div/>
     * Both tensors need to be of the same type.
     */
    void multiplyScalarTensor(Tensor scalar, Tensor tensor, Stream stream);

    /**
     * [thisTensor] = [multiplicand] * [multiplier], elementwise
     *
     * <div/>
     * Both tensors need to be of the same size and type.
     */
    void multiplyElementwise(Tensor multiplicand, Tensor multiplier, Stream stream);

    /**
     * [thisTensor] = [a] * [b] + [c], elementwise
     * <div/>
     * argument must be float or half.
     * there is no restriction on the data type of this destination tensor.
     *
     * Note: if you had been looking for a GEMM operation rather than this element-wise one, see the gemm() function or class
     * CublasMatrixMultiply.
     */
    void multiplyAccumulateElementwise(Tensor a, Tensor b, Tensor c, Stream stream);

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
     * x must be float.
     */
    void erfcinv(Tensor x, Stream stream);

    /**
     * [thisTensor] = erfcx([x]), elementwise
     * <div/>
     * Compute the scaled complementary error function that is equal to exp(x^2)*erfc(x):
     * https://www.mathworks.com/help/matlab/ref/erfcx.html x must be float. The return type may be float or half
     * x must be float.
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

    bool enableGpuDirectStorage(bool enable) {
        bool oldEnable = gpuDirectStorageEnabled;
        if (enable == oldEnable)
            return oldEnable;

        if (!this->fileName.empty()) {
            std::string attachedFileName = this->fileName;
            detachFile();
            gpuDirectStorageEnabled = enable;
            attachFile(attachedFileName, gpuDirectStorageFileOffset, fileAccessRequirement);
        } else {
            gpuDirectStorageEnabled = enable;
        }

        return oldEnable;
    }

    virtual bool isKerasCompatible(std::string &explanation) {
        explanation.clear();
        return true;
    }

    static uint32_t getThreadIdHash(uint32_t seed = 0);
    static uint64_t getThreadIdHash64(uint64_t seed = 0);

   private:
    void copyFromAsync(Tensor source, Stream copyStream, bool mustPreserveSourceValue);

    TensorPlacement placement;
    void *mem = nullptr;

    unsigned long instanceId;

    TensorDescriptor descriptor;

    bool gpuDirectStorageEnabled = false;

    bool usingExternallyManagedMemory = false;

    std::string fileName;
    int32_t fileDescriptor = 0;
    bool ownsFileDescriptor = false;
    FileAccess fileAccessRequirement;
    CUfileHandle_t gpuDirectStorageCuFileHandle = nullptr;
    off_t gpuDirectStorageFileOffset;
    size_t gpuDirectStorageSize;
    off_t gpuDirectStorageBufOffset = 0;
    ssize_t gpuDirectStorageBytesAccessed;

    bool cpuMemPinnedViaRegister = false;

    // FIXME: get rid of this override descriptor nonsense
    bool descriptorOverridden = false;
    TensorDescriptor overriddenDescriptor;

    static std::atomic<unsigned long> nextInstanceId;

    void allocateMemory(uint32_t alignmentBytes = 0);

    template <typename T>
    void launchFillValueGpuKernel(T value, T *mem, uint64_t numElements, uint32_t deviceNum, Stream stream);
    void fillGpuIdentityMatrixOnes(Stream stream);
    template <typename DATA_TYPE>
    void launchGpuFillRandom(void *mem, uint64_t numElements, double minValue, double maxValue, Stream stream);

    void overrideDescriptor(TensorDescriptor overrideDescriptor);
    void clearDescriptorOverride();

    void construct(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes, void *externallyManagedMemory);
    void copyObject(const Tensor &other);
    void destroy();
};

class CuFileInitializer {
   public:
    CuFileInitializer() {
        // First ensure cuda context has been created, via effective NOP
        cudaError_t cudaStatus;
        cudaStatus = cudaFree(nullptr);
        if (cudaStatus == cudaSuccess) {
            // Call cuFileDriverOpen() and check for errors
            CUfileError_t cuFileError = cuFileDriverOpen();
            assert(cuFileError.err == CU_FILE_SUCCESS);
        }
    }

    ~CuFileInitializer() { cuFileDriverClose(); }
};

}  // namespace ThorImplementation
