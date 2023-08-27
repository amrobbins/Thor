#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
using namespace std;

atomic<unsigned long> Tensor::nextInstanceId(1);

Tensor::Tensor() : ReferenceCounted() { usingExternallyManagedMemory = false; }

Tensor::Tensor(TensorPlacement placement, TensorDescriptor descriptor) { construct(placement, descriptor, nullptr); }

Tensor::Tensor(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory) {
    construct(placement, descriptor, externallyManagedMemory);
}

Tensor::Tensor(const Tensor &tensorInstance) {
    // implemented using operator=
    *this = tensorInstance;
}

Tensor &Tensor::operator=(const Tensor &other) {
    copyObject(other);
    return *this;
}

Tensor::~Tensor() {
    bool shouldDestroy = ReferenceCounted::removeReference();
    if (shouldDestroy)
        destroy();
}

bool Tensor::operator==(const Tensor &other) const {
    assert(!uninitialized());
    return instanceId == other.instanceId;
}

bool Tensor::operator!=(const Tensor &other) const {
    assert(!uninitialized());
    return instanceId != other.instanceId;
}

bool Tensor::operator<(const Tensor &other) const {
    assert(!uninitialized());
    return instanceId < other.instanceId;
}

void Tensor::construct(TensorPlacement placement, TensorDescriptor descriptor, void *externallyManagedMemory) {
    ReferenceCounted::initialize();

    assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU || placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(placement.getDeviceNum() >= 0);
    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU)
        assert(placement.getDeviceNum() == 0);
    else
        assert(placement.getDeviceNum() < (int)MachineEvaluator::instance().getNumGpus());
    this->placement = placement;

    this->descriptor = descriptor;
    descriptorOverridden = false;

    if (externallyManagedMemory == nullptr)
        usingExternallyManagedMemory = false;
    else
        usingExternallyManagedMemory = true;

    instanceId = nextInstanceId.fetch_add(1);

    if (usingExternallyManagedMemory) {
        mem = externallyManagedMemory;
    } else {
        allocateMemory();
    }
}

void Tensor::allocateMemory() {
    cudaError_t cudaStatus;

    unsigned long numElements = descriptor.getTotalNumElements();
    assert(numElements > 0);

    unsigned long memBytes;
    memBytes = descriptor.getArraySizeInBytes();
    // All tensors end on an 16 byte boundary so that kernels can overshoot when writing arrays of type half4 or float4
    // without risking accessing another memory block
    memBytes = (memBytes + 15) / 16;
    memBytes *= 16;

    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        cudaStatus = cudaHostAlloc(&mem, memBytes, cudaHostAllocPortable);
        assert(cudaStatus == cudaSuccess);
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(placement.getDeviceNum());
        cudaStatus = cudaMalloc(&mem, memBytes);
        if (cudaStatus != cudaSuccess) {
            printf("cudaStatus %d\n", cudaStatus);
            printf("%s\n", cudaGetErrorString(cudaStatus));
            fflush(stdout);
        }
        assert(cudaStatus == cudaSuccess);
    } else {
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
}

void Tensor::copyObject(const Tensor &other) {
    *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

    placement = other.placement;
    mem = other.mem;
    instanceId = other.instanceId;

    descriptor = other.descriptor;
    descriptorOverridden = other.descriptorOverridden;
    overriddenDescriptor = other.overriddenDescriptor;
}

void Tensor::destroy() {
    if (usingExternallyManagedMemory) {
        // NOP
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        cudaError_t cudaStatus = cudaFreeHost(mem);

        if (cudaStatus != cudaSuccess) {
            printf("cuda status %d\n", cudaStatus);
            fflush(stdout);
        }

        assert(cudaStatus == cudaSuccess);
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(placement.getDeviceNum());
        cudaError_t cudaStatus = cudaFree(mem);
        assert(cudaStatus == cudaSuccess);
    } else {
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
    mem = nullptr;
}

bool Tensor::isUsingExternallyManagedMemory() { return usingExternallyManagedMemory; }

// Use same memory, but change dimension sizes, must be exactly the same number of elements.
void Tensor::reshape(vector<unsigned long> dimensions) { descriptor.reshape(dimensions); }

// Change the dimensions of the tensor, possibly changing the amount of memory used.
// Frees the old memory and uses a new, uninitialized block of memory.
void Tensor::resize(vector<unsigned long> dimensions) {
    assert(!usingExternallyManagedMemory);

    descriptor = TensorDescriptor(descriptor.getDataType(), dimensions);
    destroy();
    allocateMemory();
}

// Stream is on either the source or dest device
void Tensor::copyFromAsync(Tensor source, Stream stream) {
    assert(!uninitialized());
    vector<int> devicesInvolved;
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(source.getPlacement().getDeviceNum());
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(getPlacement().getDeviceNum());
    if (!devicesInvolved.empty())
        assert(stream.getGpuNum() == devicesInvolved[0] || (devicesInvolved.size() == 2 && stream.getGpuNum() == devicesInvolved[1]));
    copyFromAsync(source, stream, true);
}

// If the tensor changes datatypes such that the size changes, then stream must be on the device with the larger tensor size.
// Otherwise stream may be on either device
void Tensor::moveFromAsync(Tensor source, Stream stream) {
    assert(!uninitialized());
    vector<int> devicesInvolved;
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(source.getPlacement().getDeviceNum());
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(getPlacement().getDeviceNum());
    if (!devicesInvolved.empty())
        assert(stream.getGpuNum() == devicesInvolved[0] || (devicesInvolved.size() == 2 && stream.getGpuNum() == devicesInvolved[1]));
    copyFromAsync(source, stream, false);
}

TensorDescriptor Tensor::getDescriptor() {
    assert(!uninitialized());

    if (descriptorOverridden)
        return overriddenDescriptor;
    return descriptor;
}

void Tensor::overrideDescriptor(TensorDescriptor descriptor) {
    assert(!uninitialized());

    descriptorOverridden = true;
    overriddenDescriptor = descriptor;
}

void Tensor::clearDescriptorOverride() {
    assert(!uninitialized());
    descriptorOverridden = false;
}

void *Tensor::getElement(vector<unsigned long> dimensionIndex) {
    assert(!uninitialized());

    assert(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    return getDescriptor().getChunkAddress(dimensionIndex, mem);
}

void Tensor::copyFromAsync(Tensor source, Stream copyStream, bool mustPreserveSourceValue) {
    assert(!uninitialized());

    if (source.getTensorId() == getTensorId() && source.getDescriptor().getDataType() == getDescriptor().getDataType()) {
        return;
    }

    cudaError_t cudaStatus;
    assert(copyStream.isInitialized());

    // must have the same number of elements
    TensorDescriptor sourceDescriptor = source.getDescriptor();
    TensorDescriptor destDescriptor = getDescriptor();
    if (sourceDescriptor.getTotalNumElements() != destDescriptor.getTotalNumElements()) {
        printf("Error: total number of elements does not match when copying tensors.\n source dimensions %s\n dest dimensions %s\n",
               source.dimensionsToString().c_str(),
               dimensionsToString().c_str());
        fflush(stdout);
    }
    assert(sourceDescriptor.getTotalNumElements() == destDescriptor.getTotalNumElements());

    int sourceDeviceNum = source.placement.getDeviceNum();
    int destDeviceNum = placement.getDeviceNum();

    // Handle across device conversions:
    //
    // If the destination data type is larger than the source data type, then this is always supported.
    //      - The data is copied to the dest device and then up converted in place.
    //
    // If the destination data type is smaller then the source data type, then this is only supported when mustPreserveSourceValue is false.
    //      - In this case an inplace down conversion is performed in source memory and the converted mem is copied to the dest device.
    if (sourceDescriptor.getDataType() != destDescriptor.getDataType() && source.placement != placement) {
        if (sourceDescriptor.getArraySizeInBytes() <= destDescriptor.getArraySizeInBytes()) {
            if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU)
                assert(placement.getDeviceNum() == copyStream.getGpuNum());

            Tensor meWithSourceDataType = *this;
            meWithSourceDataType.overrideDescriptor(sourceDescriptor);
            meWithSourceDataType.copyFromAsync(source, copyStream, mustPreserveSourceValue);

            TypeConverter::convertType(
                mem,
                mem,
                sourceDescriptor.getDataType(),
                destDescriptor.getDataType(),
                sourceDescriptor.getTotalNumElements(),
                copyStream,
                placement.getMemDevice() == TensorPlacement::MemDevices::CPU ? MachineEvaluator::CPU_DEVICE_NUM : destDeviceNum);

            return;
        } else {
            assert(!mustPreserveSourceValue);

            if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU)
                assert(source.placement.getDeviceNum() == copyStream.getGpuNum());

            TypeConverter::convertType(
                source.mem,
                source.mem,
                sourceDescriptor.getDataType(),
                destDescriptor.getDataType(),
                sourceDescriptor.getTotalNumElements(),
                copyStream,
                source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU ? MachineEvaluator::CPU_DEVICE_NUM : sourceDeviceNum);

            Tensor sourceWithMyDataType = source;
            sourceWithMyDataType.overrideDescriptor(destDescriptor);
            copyFromAsync(sourceWithMyDataType, copyStream, mustPreserveSourceValue);

            if (source.placement.getMemDevice() != TensorPlacement::MemDevices::CPU && copyStream.getGpuNum() != sourceDeviceNum)
                copyStream.waitEvent(copyStream.putEvent());

            return;
        }
    }

    if (source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU &&
        placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        ScopedGpu scopedGpu(0);  // CPU local stream belongs to gpu 0

        if (sourceDescriptor.getDataType() == destDescriptor.getDataType()) {
            cudaStatus =
                cudaMemcpyAsync(mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyHostToHost, copyStream.getStream());
            assert(cudaStatus == cudaSuccess);
        } else {
            // Convert between data types on cpu.
            // Note that this may be an in-place conversion
            TypeConverter::convertType(source.mem,
                                       mem,
                                       sourceDescriptor.getDataType(),
                                       destDescriptor.getDataType(),
                                       destDescriptor.getTotalNumElements(),
                                       copyStream,
                                       MachineEvaluator::CPU_DEVICE_NUM);
        }
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(destDeviceNum);

        cudaStatus =
            cudaMemcpyAsync(mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyHostToDevice, copyStream.getStream());
        assert(cudaStatus == cudaSuccess);
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        ScopedGpu scopedGpu(sourceDeviceNum);

        cudaStatus =
            cudaMemcpyAsync(mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyDeviceToHost, copyStream.getStream());
        assert(cudaStatus == cudaSuccess);
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        if (destDeviceNum == sourceDeviceNum) {
            // Local copy
            ScopedGpu scopedGpu(destDeviceNum);

            if (sourceDescriptor.getDataType() == destDescriptor.getDataType()) {
                cudaStatus = cudaMemcpyAsync(
                    mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyDeviceToDevice, copyStream.getStream());
                assert(cudaStatus == cudaSuccess);
            } else {
                // Convert between data types on device.
                // Note that this may be an in-place conversion
                TypeConverter::convertType(source.mem,
                                           mem,
                                           sourceDescriptor.getDataType(),
                                           destDescriptor.getDataType(),
                                           sourceDescriptor.getTotalNumElements(),
                                           copyStream,
                                           sourceDeviceNum);
            }
        } else {
            // Cross device copy
            ScopedGpu scopedGpu(destDeviceNum);

            cudaStatus = cudaMemcpyPeerAsync(
                mem, destDeviceNum, source.mem, sourceDeviceNum, sourceDescriptor.getArraySizeInBytes(), copyStream.getStream());
            assert(cudaStatus == cudaSuccess);
        }
    } else {
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
}

void Tensor::memset(int8_t value, uint64_t numElements) {
    uint64_t numBytes;
    if (numElements == 0) {
        numBytes = getArraySizeInBytes();
    } else {
        // If you need to set part of the last packed boolean to 0, you will need 2 calls to memset, one for zeros one for last value.
        if (getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            assert(numElements % 8 == 0);
            numBytes = numElements / 8;
        } else {
            numBytes = numElements * (getArraySizeInBytes() / getTotalNumElements());
        }
    }

    if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(placement.getDeviceNum());
        cudaError_t cudaStatus;
        cudaStatus = cudaMemset(mem, 0, numBytes);
        assert(cudaStatus == cudaSuccess);
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        // invoke global memset instead of member function memset
        ::memset(mem, value, numBytes);
    } else {
        assert(false);
    }
}

void Tensor::memsetAsync(Stream stream, int8_t value, uint64_t numElements) {
    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    uint64_t numBytes;
    if (numElements == 0) {
        numBytes = getArraySizeInBytes();
    } else {
        // If you need to set part of the last packed boolean to 0, you will need 2 calls to memset, one for zeros one for last value.
        if (getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            assert(numElements % 8 == 0);
            numBytes = numElements / 8;
        } else {
            numBytes = numElements * (getArraySizeInBytes() / getTotalNumElements());
        }
    }

    ScopedGpu scopedGpu(placement.getDeviceNum());
    cudaError_t cudaStatus;
    cudaStatus = cudaMemsetAsync(mem, value, numBytes, stream);
    assert(cudaStatus == cudaSuccess);
}

string Tensor::dimensionsToString() {
    string s = "[";
    vector<uint64_t> dimensions = getDimensions();
    for (uint32_t i = 0; i < dimensions.size(); ++i) {
        if (i > 0)
            s += ", ";
        s += to_string(dimensions[i]);
    }
    s += "]";
    return s;
}

// setValues is intended as a test helper to easily populate an entire tensor
// It is less efficent than working with tensor memory directly since it uses non-pinned cpu memory and is not meant to be used
// in performance critical code.
template <typename T>
void Tensor::setValues(vector<T> values, Stream stream) {
    assert(values.size() == getTotalNumElements());
    assert(!values.empty());

    if (is_same<T, half>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::FP32);
    else if (is_same<T, double>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::FP64);
    else if (is_same<T, int8_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT8);
    else if (is_same<T, int16_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT16);
    else if (is_same<T, int32_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT32);
    else if (is_same<T, int64_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT64);
    else if (is_same<T, uint8_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT8 ||
               descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN);
    else if (is_same<T, uint16_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT16);
    else if (is_same<T, uint32_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT32);
    else if (is_same<T, uint64_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT64);
    else
        assert(false);

    Tensor tempTensor(TensorPlacement::MemDevices::CPU, descriptor, values.data());
    this->copyFromAsync(tempTensor, stream);
    stream.synchronize();
}

template <typename T>
void Tensor::loadValuesIntoVector(std::vector<T> &values, Stream stream) {
    if (is_same<T, half>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::FP32);
    else if (is_same<T, double>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::FP64);
    else if (is_same<T, int8_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT8);
    else if (is_same<T, int16_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT16);
    else if (is_same<T, int32_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT32);
    else if (is_same<T, int64_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::INT64);
    else if (is_same<T, uint8_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT8 ||
               descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN);
    else if (is_same<T, uint16_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT16);
    else if (is_same<T, uint32_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT32);
    else if (is_same<T, uint64_t>::value)
        assert(descriptor.getDataType() == TensorDescriptor::DataType::UINT64);
    else
        assert(false);

    values.clear();
    Tensor tempTensor(TensorPlacement::MemDevices::CPU, descriptor);
    tempTensor.copyFromAsync(*this, stream);
    stream.synchronize();

    T *mem = (T *)(tempTensor.getMemPtr());
    uint64_t numElements = getTotalNumElements();
    for (uint64_t i = 0; i < numElements; ++i)
        values.push_back(mem[i]);
}

Tensor Tensor::transposeMatrix(Stream stream) {
    vector<uint64_t> dimensions = getDimensions();
    // Generally the transpose of a higher order tensor would be any permutation of the tensor's dimensions, in that case the particular
    // permutation would also need to be specified. I'm not doing that now and unless some need arises I probably won't implement that.
    assert(dimensions.size() == 2);
    vector<uint64_t> transposedDimensions;
    transposedDimensions.push_back(dimensions[1]);
    transposedDimensions.push_back(dimensions[0]);
    Tensor transposedTensor = clone(transposedDimensions);

    if (getDataType() == TensorDescriptor::DataType::FP16) {
        matrixTranspose((half *)transposedTensor.getMemPtr(), (half *)getMemPtr(), dimensions[0], dimensions[1], stream);
    } else if (getDataType() == TensorDescriptor::DataType::FP32) {
        matrixTranspose((float *)transposedTensor.getMemPtr(), (float *)getMemPtr(), dimensions[0], dimensions[1], stream);
    } else {
        assert(false);  // TODO
    }

    return transposedTensor;
}

void Tensor::transposeSquareMatrixInPlace(Stream stream) {
    vector<uint64_t> dimensions = getDimensions();
    // Generally the transpose of a higher order tensor would be any permutation of the tensor's dimensions, in that case the particular
    // permutation would also need to be specified. I'm not doing that now and unless some need arises I probably won't implement that.
    assert(dimensions.size() == 2);
    assert(dimensions[0] == dimensions[1]);

    if (getDataType() == TensorDescriptor::DataType::FP16) {
        matrixTransposeSquare((half *)getMemPtr(), (half *)getMemPtr(), dimensions[0], stream);
    } else if (getDataType() == TensorDescriptor::DataType::FP32) {
        matrixTransposeSquare((float *)getMemPtr(), (float *)getMemPtr(), dimensions[0], stream);
    } else {
        assert(false);  // TODO
    }
}

template void Tensor::setValues<half>(vector<half> values, Stream stream);
template void Tensor::setValues<float>(vector<float> values, Stream stream);
template void Tensor::setValues<double>(vector<double> values, Stream stream);
template void Tensor::setValues<int8_t>(vector<int8_t> values, Stream stream);
template void Tensor::setValues<int16_t>(vector<int16_t> values, Stream stream);
template void Tensor::setValues<int32_t>(vector<int32_t> values, Stream stream);
template void Tensor::setValues<int64_t>(vector<int64_t> values, Stream stream);
template void Tensor::setValues<uint8_t>(vector<uint8_t> values, Stream stream);
template void Tensor::setValues<uint16_t>(vector<uint16_t> values, Stream stream);
template void Tensor::setValues<uint32_t>(vector<uint32_t> values, Stream stream);
template void Tensor::setValues<uint64_t>(vector<uint64_t> values, Stream stream);

template void Tensor::loadValuesIntoVector<half>(vector<half> &values, Stream stream);
template void Tensor::loadValuesIntoVector<float>(vector<float> &values, Stream stream);
template void Tensor::loadValuesIntoVector<double>(vector<double> &values, Stream stream);
template void Tensor::loadValuesIntoVector<int8_t>(vector<int8_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<int16_t>(vector<int16_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<int32_t>(vector<int32_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<int64_t>(vector<int64_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<uint8_t>(vector<uint8_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<uint16_t>(vector<uint16_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<uint32_t>(vector<uint32_t> &values, Stream stream);
template void Tensor::loadValuesIntoVector<uint64_t>(vector<uint64_t> &values, Stream stream);
