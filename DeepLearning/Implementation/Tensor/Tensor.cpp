#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "Thor.h"

using namespace ThorImplementation;

atomic<unsigned long> Tensor::nextInstanceId(1);

Tensor::Tensor() : ReferenceCounted() {}

Tensor::Tensor(TensorPlacement placement, TensorDescriptor descriptor) { construct(placement, descriptor); }

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

void Tensor::construct(TensorPlacement placement, TensorDescriptor descriptor) {
    ReferenceCounted::initialize();

    cudaError_t cudaStatus;

    assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU || placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(placement.getDeviceNum() >= 0);
    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU)
        assert(placement.getDeviceNum() == 0);
    else
        assert(placement.getDeviceNum() < (int)MachineEvaluator::instance().getNumGpus());
    this->placement = placement;

    this->descriptor = descriptor;
    descriptorOverridden = false;

    instanceId = nextInstanceId.fetch_add(1);

    unsigned long numElements = descriptor.getTotalNumElements();
    assert(numElements > 0);

    unsigned long memBytes;
    memBytes = descriptor.getArraySizeInBytes();

    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        cudaStatus = cudaHostAlloc(&mem, memBytes, cudaHostAllocPortable);
        assert(cudaStatus == cudaSuccess);
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(placement.getDeviceNum());

        cudaStatus = cudaMalloc(&mem, memBytes);
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
    cudaError_t cudaStatus;

    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        cudaStatus = cudaFreeHost(mem);
        assert(cudaStatus == cudaSuccess);
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(placement.getDeviceNum());

        cudaStatus = cudaFree(mem);
        assert(cudaStatus == cudaSuccess);
        mem = NULL;
    } else {
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
}

// Use same memory, but change dimension sizes, must be exactly the same number of elements.
void Tensor::reshape(vector<unsigned long> dimensions) { descriptor.reshape(dimensions); }

void Tensor::copyFromAsync(Tensor source, Stream stream) {
    assert(!uninitialized());
    copyFromAsync(source, stream, true);
}
void Tensor::copyFromAsync(DistributedTensor source, Stream stream) {
    assert(!uninitialized());
    copyFromAsync(source, stream, true);
}

void Tensor::moveFromAsync(Tensor source, Stream stream) {
    assert(!uninitialized());
    copyFromAsync(source, stream, false);
}
void Tensor::moveFromAsync(DistributedTensor source, Stream stream) {
    assert(!uninitialized());
    copyFromAsync(source, stream, false);
}

// The following function variants return an event that indicates that the copying is finished when
// cudaStreamWaitEvent() is called on this event.
Event Tensor::copyFromAsync(Tensor source, Event startEvent) {
    assert(!uninitialized());
    return copyFromAsync(source, startEvent, true);
}
Event Tensor::copyFromAsync(DistributedTensor source, Event startEvent) {
    assert(!uninitialized());
    return copyFromAsync(source, startEvent, true);
}

Event Tensor::moveFromAsync(Tensor source, Event startEvent) {
    assert(!uninitialized());
    return copyFromAsync(source, startEvent, false);
}
Event Tensor::moveFromAsync(DistributedTensor source, Event startEvent) {
    assert(!uninitialized());
    return copyFromAsync(source, startEvent, false);
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

// The stream will wait for the copy to finish,
// but if you have another stream that you want to wait also, you can use the returned Event to do that.
void Tensor::copyFromAsync(Tensor source, Stream stream, bool mustPreserveSourceValue) {
    assert(!uninitialized());

    Event finishedEvent;

    finishedEvent = copyFromAsync(source, stream.putEvent(), mustPreserveSourceValue);
    stream.waitEvent(finishedEvent);
}

// Note: startEvent is on the source device, finished event is on the dest device, copyStream is on the dest device
Event Tensor::copyFromAsync(Tensor source, Event startEvent, bool mustPreserveSourceValue) {
    assert(!uninitialized());

    cudaError_t cudaStatus;
    Stream copyStream;
    Event finishedEvent;

    if (source.getTensorId() == getTensorId() && source.getDescriptor().getDataType() == getDescriptor().getDataType()) {
        return startEvent;
    }

    // must have the same number of elements
    TensorDescriptor sourceDescriptor = source.getDescriptor();
    TensorDescriptor destDescriptor = getDescriptor();
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
            Tensor meWithSourceDataType = *this;
            meWithSourceDataType.overrideDescriptor(sourceDescriptor);
            Event finishedCopyingEvent = meWithSourceDataType.copyFromAsync(source, startEvent, mustPreserveSourceValue);

            if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU)
                copyStream = MachineEvaluator::instance().getCopyStreamLocal(MachineEvaluator::CPU_DEVICE_NUM);
            else
                copyStream = MachineEvaluator::instance().getCopyStreamLocal(destDeviceNum);

            copyStream.waitEvent(finishedCopyingEvent);
            TypeConverter::convertType(
                mem,
                mem,
                sourceDescriptor.getDataType(),
                destDescriptor.getDataType(),
                sourceDescriptor.getTotalNumElements(),
                copyStream,
                placement.getMemDevice() == TensorPlacement::MemDevices::CPU ? MachineEvaluator::CPU_DEVICE_NUM : destDeviceNum);

            return copyStream.putEvent();
        } else {
            assert(!mustPreserveSourceValue);

            if (source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU)
                copyStream = MachineEvaluator::instance().getCopyStreamLocal(MachineEvaluator::CPU_DEVICE_NUM);
            else
                copyStream = MachineEvaluator::instance().getCopyStreamLocal(sourceDeviceNum);

            copyStream.waitEvent(startEvent);
            TypeConverter::convertType(
                source.mem,
                source.mem,
                sourceDescriptor.getDataType(),
                destDescriptor.getDataType(),
                sourceDescriptor.getTotalNumElements(),
                copyStream,
                source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU ? MachineEvaluator::CPU_DEVICE_NUM : sourceDeviceNum);
            Event finishedConversionEvent = copyStream.putEvent();

            Tensor sourceWithMyDataType = source;
            sourceWithMyDataType.overrideDescriptor(destDescriptor);

            return copyFromAsync(sourceWithMyDataType, finishedConversionEvent, mustPreserveSourceValue);
        }
    }

    if (source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU &&
        placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        ScopedGpu scopedGpu(0);  // CPU local stream belongs to gpu 0

        copyStream = MachineEvaluator::instance().getCopyStreamLocal(MachineEvaluator::CPU_DEVICE_NUM);
        copyStream.waitEvent(startEvent);

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

        finishedEvent = copyStream.putEvent();
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(destDeviceNum);

        copyStream = MachineEvaluator::instance().getCopyStreamFromCpu(destDeviceNum);
        copyStream.waitEvent(startEvent);

        cudaStatus =
            cudaMemcpyAsync(mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyHostToDevice, copyStream.getStream());
        assert(cudaStatus == cudaSuccess);

        finishedEvent = copyStream.putEvent();
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        ScopedGpu scopedGpu(sourceDeviceNum);

        copyStream = MachineEvaluator::instance().getCopyStreamToCpu(sourceDeviceNum);
        copyStream.waitEvent(startEvent);

        cudaStatus =
            cudaMemcpyAsync(mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyDeviceToHost, copyStream.getStream());
        assert(cudaStatus == cudaSuccess);

        finishedEvent = copyStream.putEvent();
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        int destBusId = MachineEvaluator::instance().getGpuPciBusId(destDeviceNum);
        int sourceBusId = MachineEvaluator::instance().getGpuPciBusId(sourceDeviceNum);
        if (sourceBusId < destBusId) {
            copyStream = MachineEvaluator::instance().getCopyStreamFromLower(destDeviceNum);
        } else if (sourceBusId > destBusId) {
            copyStream = MachineEvaluator::instance().getCopyStreamFromHigher(destDeviceNum);
        } else {
            // source and dest is the same gpu:
            copyStream = MachineEvaluator::instance().getCopyStreamLocal(destDeviceNum);
        }
        copyStream.waitEvent(startEvent);

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

            finishedEvent = copyStream.putEvent();
        } else {
            // Cross device copy
            ScopedGpu scopedGpu(destDeviceNum);

            cudaStatus = cudaMemcpyPeerAsync(
                mem, destDeviceNum, source.mem, sourceDeviceNum, sourceDescriptor.getArraySizeInBytes(), copyStream.getStream());
            assert(cudaStatus == cudaSuccess);

            finishedEvent = copyStream.putEvent();
        }
    } else {
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }

    return finishedEvent;
}

void Tensor::copyFromAsync(DistributedTensor source, Stream stream, bool mustPreserveSourceValue) {
    assert(!uninitialized());

    Event finishedEvent;

    finishedEvent = copyFromAsync(source, stream.putEvent(), mustPreserveSourceValue);
    stream.waitEvent(finishedEvent);
}

Event Tensor::copyFromAsync(DistributedTensor source, Event startEvent, bool mustPreserveSourceValue) {
    assert(!uninitialized());

    assert(source.getNumInstances() > 0);

    return copyFromAsync(source.getNearestInstance(*this), startEvent, mustPreserveSourceValue);
}
