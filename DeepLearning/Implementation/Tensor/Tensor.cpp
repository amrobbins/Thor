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