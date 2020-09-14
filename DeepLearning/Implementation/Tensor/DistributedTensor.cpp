#include "DeepLearning/Implementation/Tensor/DistributedTensor.h"

#include "Thor.h"

// FIXME : get rid of the copyFromAsync duplication,
//        handle the minor difference some other way.

atomic<unsigned long> DistributedTensor::nextTensorId(1);

DistributedTensor::DistributedTensor() : ReferenceCounted() {
    tensorMutex = nullptr;
    instances = nullptr;
}

DistributedTensor::DistributedTensor(TensorDescriptor descriptor) { construct(descriptor); }

DistributedTensor::DistributedTensor(const DistributedTensor &tensor) {
    // implemented using operator=
    *this = tensor;
}

DistributedTensor &DistributedTensor::operator=(const DistributedTensor &other) {
    copyObject(other);
    return *this;
}

DistributedTensor::~DistributedTensor() {
    bool shouldDestroy = ReferenceCounted::removeReference();
    if (shouldDestroy)
        destroy();
}

void DistributedTensor::construct(TensorDescriptor descriptor) {
    ReferenceCounted::initialize();

    assert(descriptor.getNumDimensions() > 0);

    tensorMutex = new recursive_mutex();
    instances = new unordered_map<unsigned long, Tensor>();

    this->descriptor = descriptor;

    distributedTensorId = nextTensorId.fetch_add(1);
}

void DistributedTensor::copyObject(const DistributedTensor &other) {
    *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

    descriptor = other.descriptor;
    instances = other.instances;
    tensorMutex = other.tensorMutex;
    distributedTensorId = other.distributedTensorId;
}

void DistributedTensor::destroy() {
    delete tensorMutex;
    delete instances;
}

bool DistributedTensor::operator==(const DistributedTensor &other) const {
    assert(!uninitialized());
    return getDistributedTensorId() == other.getDistributedTensorId();
}
bool DistributedTensor::operator!=(const DistributedTensor &other) const {
    assert(!uninitialized());
    return getDistributedTensorId() != other.getDistributedTensorId();
}

Tensor DistributedTensor::getAnyInstance() {
    assert(!uninitialized());

    assert(!instances->empty());
    return (instances->begin())->second;
}

void DistributedTensor::copyFromAsync(DistributedTensor source, Stream stream) {
    assert(!uninitialized());

    // deviceNum -> list of tensor instances on that device
    int deviceNum;
    map<int, Tensor> populatedInstancePerDevice;
    map<int, vector<Tensor>> unpopulatedInstancesPerDevice;

    if (getDistributedTensorId() == source.getDistributedTensorId())
        return;

    for (auto itSource = source.instances->begin(); itSource != source.instances->end(); ++itSource) {
        Tensor &sourceTensorInstance = itSource->second;
        if (sourceTensorInstance.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU)
            deviceNum = MachineEvaluator::CPU_DEVICE_NUM;
        else
            deviceNum = sourceTensorInstance.getPlacement().getDeviceNum();
        if (populatedInstancePerDevice.count(deviceNum) == 0)
            populatedInstancePerDevice[deviceNum] = sourceTensorInstance;
    }

    for (auto itDest = instances->begin(); itDest != instances->end(); ++itDest) {
        Tensor &destTensorInstance = itDest->second;
        if (destTensorInstance.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU)
            deviceNum = MachineEvaluator::CPU_DEVICE_NUM;
        else
            deviceNum = destTensorInstance.getPlacement().getDeviceNum();
        unpopulatedInstancesPerDevice[deviceNum].push_back(destTensorInstance);
    }
    if (unpopulatedInstancesPerDevice.empty())
        return;

    // DataType conversion is legal during copy if 1. the conversion is to a larger data type or 2. the two tensors have
    // at least one instance on the same device DataType conversion to a floating point type is performed on only one
    // type of device, to avoid differences in floating point values between multiple instances of the same tensor, GPU
    // is the preferred device to do the conversion and will be used when possible.
    set<TensorDescriptor::DataType> floatingPointDataTypes;
    floatingPointDataTypes.insert(TensorDescriptor::DataType::FP16);
    floatingPointDataTypes.insert(TensorDescriptor::DataType::FP32);
    floatingPointDataTypes.insert(TensorDescriptor::DataType::FP64);
    if (descriptor.getDataType() != source.getDescriptor().getDataType()) {
        if (descriptor.getNumBytesPerElement() < source.getDescriptor().getNumBytesPerElement()) {
            // FIXME: THIS CASE
            // destination tensor is a smaller dataType, so convert before transmitting it.

            map<int, Tensor> convertedInstancePerDevice;
            map<int, Event> populatedEventPerDevice;

            peformOnGpuConversions(
                populatedInstancePerDevice, unpopulatedInstancesPerDevice, stream, convertedInstancePerDevice, populatedEventPerDevice);

            assert(!convertedInstancePerDevice.empty());
            populatedInstancePerDevice = convertedInstancePerDevice;

            map<int, Event> deviceFinishedEvents =
                copyFromAsyncImpl(populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);

            for (auto entry : deviceFinishedEvents) {
                Event deviceFinishedEvent = entry.second;
                stream.waitEvent(deviceFinishedEvent);
            }
        } else {
            // destination tensor is a larger dataType, so transmit before in-place up conversion.
            map<int, Tensor> stagingInstancePerDevice;
            map<int, Tensor> upconvertedInstancePerDevice;
            for (auto it = unpopulatedInstancesPerDevice.begin(); it != unpopulatedInstancesPerDevice.end(); ++it) {
                if (it->first == MachineEvaluator::CPU_DEVICE_NUM && floatingPointDataTypes.count(descriptor.getDataType()) == 1 &&
                    floatingPointDataTypes.count(descriptor.getDataType()) == 1)
                    continue;
                stagingInstancePerDevice[it->first] = it->second.front();
                stagingInstancePerDevice[it->first].overrideDescriptor(source.getDescriptor());
            }

            if (stagingInstancePerDevice.empty()) {
                // If no staging instance on any gpu, meaning all destination instances are on CPU, so do type
                // conversion on CPU

                assert(unpopulatedInstancesPerDevice.count(MachineEvaluator::CPU_DEVICE_NUM) == 1);
                assert(!unpopulatedInstancesPerDevice[MachineEvaluator::CPU_DEVICE_NUM].empty());

                Tensor cpuInstance = unpopulatedInstancesPerDevice[MachineEvaluator::CPU_DEVICE_NUM].front();
                Tensor instanceLargeDataType = cpuInstance;
                Tensor instanceSmallDataType = cpuInstance;
                instanceSmallDataType.overrideDescriptor(source.getDescriptor());

                instanceSmallDataType.copyFromAsync(source, stream);
                instanceLargeDataType.copyFromAsync(instanceSmallDataType, stream);
                copyFromAsync(instanceLargeDataType, stream);
            } else {
                // There is a staging instance on a GPU, so do all type conversions on GPU, copy back to CPU if needed

                // Now I have one instance on each GPU that has any instance in stagingInstancePerDevice,
                // copy into all staging instances
                map<int, Event> populatedEventPerDevice;
                for (auto it = populatedInstancePerDevice.begin(); it != populatedInstancePerDevice.end(); ++it) {
                    int deviceNum = it->first;
                    populatedEventPerDevice[deviceNum] = stream.putEvent();
                }
                map<int, Event> deviceStagingInstanceReadyEvents =
                    copyFromAsyncImpl(populatedInstancePerDevice, stagingInstancePerDevice, populatedEventPerDevice);

                // Now I have one populated instance on every dest device, upconvert that instance in-place
                for (auto it = deviceStagingInstanceReadyEvents.begin(); it != deviceStagingInstanceReadyEvents.end(); ++it) {
                    // FIXME: This part should be part of Tensor.copyFromAsync(Tensor...)
                    int deviceNum = it->first;
                    Event stagingInstanceCopiedEvent = it->second;
                    Tensor instanceSmallDataType = stagingInstancePerDevice[deviceNum];
                    Tensor instanceLargeDataType = stagingInstancePerDevice[deviceNum];
                    instanceLargeDataType.clearDescriptorOverride();

                    // Up convert
                    Event stagingInstanceConvertedEvent =
                        instanceLargeDataType.copyFromAsync(instanceSmallDataType, stagingInstanceCopiedEvent);
                    deviceStagingInstanceReadyEvents[deviceNum] = stagingInstanceConvertedEvent;
                    upconvertedInstancePerDevice[deviceNum] = instanceLargeDataType;
                }
                // Next, copy a converted instance to the CPU
                // (this is for 1. to prefer GPU for floating point conversions and 2. to avoid converting between
                // floating points values on different types of device)
                if (unpopulatedInstancesPerDevice.count(MachineEvaluator::CPU_DEVICE_NUM) == 1 &&
                    upconvertedInstancePerDevice.count(MachineEvaluator::CPU_DEVICE_NUM) == 0) {
                    auto it = upconvertedInstancePerDevice.begin();
                    int copyFromDeviceNum = it->first;
                    Tensor cpuInstance = unpopulatedInstancesPerDevice[MachineEvaluator::CPU_DEVICE_NUM].back();
                    if (unpopulatedInstancesPerDevice[MachineEvaluator::CPU_DEVICE_NUM].empty()) {
                        unpopulatedInstancesPerDevice[MachineEvaluator::CPU_DEVICE_NUM].pop_back();
                        unpopulatedInstancesPerDevice.erase(MachineEvaluator::CPU_DEVICE_NUM);
                    }

                    Event stagingInstanceConvertedEvent = cpuInstance.copyFromAsync(upconvertedInstancePerDevice[copyFromDeviceNum],
                                                                                    deviceStagingInstanceReadyEvents[copyFromDeviceNum]);
                    deviceStagingInstanceReadyEvents[MachineEvaluator::CPU_DEVICE_NUM] = stagingInstanceConvertedEvent;
                    upconvertedInstancePerDevice[MachineEvaluator::CPU_DEVICE_NUM] = cpuInstance;
                }

                map<int, Event> deviceFinishedEvents =
                    copyFromAsyncImpl(upconvertedInstancePerDevice, unpopulatedInstancesPerDevice, deviceStagingInstanceReadyEvents);
                for (auto entry : deviceFinishedEvents) {
                    Event deviceFinishedEvent = entry.second;
                    stream.waitEvent(deviceFinishedEvent);
                }
                // Last, perform local copies from the upconverted instances to fill all remaining instances
                // for (auto it = unpopulatedInstancesPerDevice.begin(); it != unpopulatedInstancesPerDevice.end();
                // ++it) {
                //    int deviceNum = it->first;
                //    localDeviceCopy(deviceNum, upconvertedInstancePerDevice, unpopulatedInstancesPerDevice,
                //    deviceStagingInstanceReadyEvents);
                //}
            }
        }
    } else {
        copyFromAsyncImpl(populatedInstancePerDevice, unpopulatedInstancesPerDevice, stream);
    }
}

// Copies the data from any size-matched tensor instance into all instances of this tensor
// In the case that the source tensor instance belongs to this tensor, all other tensor instances are updated to match
// the source instance, and the source instance is not touched.
void DistributedTensor::copyFromAsync(Tensor source, Stream stream) {
    assert(!uninitialized());

    // deviceNum -> list of tensor instances on that device
    map<int, Tensor> populatedInstancePerDevice;
    map<int, vector<Tensor>> unpopulatedInstancesPerDevice;

    int deviceNum;
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU)
        deviceNum = MachineEvaluator::CPU_DEVICE_NUM;
    else
        deviceNum = source.getPlacement().getDeviceNum();
    populatedInstancePerDevice[deviceNum] = source;

    for (auto itDest = instances->begin(); itDest != instances->end(); ++itDest) {
        Tensor &destTensorInstance = itDest->second;
        if (source.getTensorId() != destTensorInstance.getTensorId()) {
            if (destTensorInstance.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU)
                deviceNum = MachineEvaluator::CPU_DEVICE_NUM;
            else
                deviceNum = destTensorInstance.getPlacement().getDeviceNum();
            unpopulatedInstancesPerDevice[deviceNum].push_back(destTensorInstance);
        }
    }
    if (unpopulatedInstancesPerDevice.empty())
        return;

    // DataType conversion is only legal during copy if the two tensors have at least one instance on the same GPU
    // DataType conversion is not performed on CPU, because it can quickly become the bottleneck, so avoiding that
    // issue.
    if (descriptor.getDataType() != source.getDescriptor().getDataType()) {
        map<int, Tensor> convertedInstancePerDevice;
        map<int, Event> populatedEventPerDevice;

        peformOnGpuConversions(
            populatedInstancePerDevice, unpopulatedInstancesPerDevice, stream, convertedInstancePerDevice, populatedEventPerDevice);

        assert(!convertedInstancePerDevice.empty());
        populatedInstancePerDevice = convertedInstancePerDevice;

        map<int, Event> deviceFinishedEvents =
            copyFromAsyncImpl(populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);

        for (auto entry : deviceFinishedEvents) {
            Event deviceFinishedEvent = entry.second;
            stream.waitEvent(deviceFinishedEvent);
        }
    } else {
        copyFromAsyncImpl(populatedInstancePerDevice, unpopulatedInstancesPerDevice, stream);
    }
}

void DistributedTensor::peformOnGpuConversions(map<int, Tensor> &populatedInstancePerDevice,
                                               map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                               Stream stream,
                                               map<int, Tensor> &convertedInstancePerDevice,
                                               map<int, Event> &populatedEventPerDevice) {
    assert(!uninitialized());

    // To implement this type conversion with copy, I will convert one unpopulated tensor instance on every gpu that has
    // a populated instance, then I will broadcast these converted instances to every dest instance that is still
    // unpopulated. Note that the copyFromAsync(...) function actually performs the conversion, and it requires that the
    // source and dest instances are on the same device.
    convertedInstancePerDevice.clear();
    populatedEventPerDevice.clear();

    for (auto it = populatedInstancePerDevice.begin(); it != populatedInstancePerDevice.end(); ++it) {
        int deviceNum = it->first;
        // If the device is actually the CPU, ignore that instance since conversion will not be done on CPU.
        if (deviceNum == MachineEvaluator::CPU_DEVICE_NUM)
            continue;
        if (unpopulatedInstancesPerDevice.count(deviceNum) == 1) {
            // There is a populated instance on this GPU
            convertedInstancePerDevice[deviceNum] = unpopulatedInstancesPerDevice[deviceNum].back();
            unpopulatedInstancesPerDevice[deviceNum].pop_back();
            if (unpopulatedInstancesPerDevice[deviceNum].empty())
                unpopulatedInstancesPerDevice.erase(deviceNum);
            populatedEventPerDevice[deviceNum] =
                convertedInstancePerDevice[deviceNum].copyFromAsync(populatedInstancePerDevice[deviceNum], stream.putEvent());
        }
    }
}

void DistributedTensor::crossDeviceCopy(int copyToGpuNum,
                                        int copyFromGpuNum,
                                        map<int, Tensor> &populatedInstancePerDevice,
                                        map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                        map<int, Event> &populatedEventPerDevice) {
    assert(!uninitialized());

    // printf("copyTo %d copyFrom %d\n", copyToGpuNum, copyFromGpuNum);
    Event finishedEvent = unpopulatedInstancesPerDevice[copyToGpuNum].back().copyFromAsync(populatedInstancePerDevice[copyFromGpuNum],
                                                                                           populatedEventPerDevice[copyFromGpuNum]);
    populatedEventPerDevice[copyToGpuNum] = finishedEvent;
    populatedInstancePerDevice[copyToGpuNum] = unpopulatedInstancesPerDevice[copyToGpuNum].back();

    unpopulatedInstancesPerDevice[copyToGpuNum].pop_back();
    if (unpopulatedInstancesPerDevice[copyToGpuNum].empty())
        unpopulatedInstancesPerDevice.erase(copyToGpuNum);
}

void DistributedTensor::localDeviceCopy(int gpuNum,
                                        map<int, Tensor> &populatedInstancePerDevice,
                                        map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                        map<int, Event> &populatedEventPerDevice) {
    assert(!uninitialized());

    if (unpopulatedInstancesPerDevice.count(gpuNum) == 0)
        return;
    assert(populatedInstancePerDevice.count(gpuNum) == 1);
    assert(populatedEventPerDevice.count(gpuNum) == 1);

    for (unsigned int i = 0; i < unpopulatedInstancesPerDevice[gpuNum].size(); ++i) {
        Event finishedEvent =
            unpopulatedInstancesPerDevice[gpuNum][i].copyFromAsync(populatedInstancePerDevice[gpuNum], populatedEventPerDevice[gpuNum]);
        populatedEventPerDevice[gpuNum] = finishedEvent;
    }
}

map<int, Event> DistributedTensor::copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                                                     map<int, Tensor> &unpopulatedInstancePerDevice,
                                                     map<int, Event> &populatedEventPerDevice) {
    assert(!uninitialized());

    map<int, vector<Tensor>> unpopulatedInstancesPerDevice;
    for (auto it = unpopulatedInstancePerDevice.begin(); it != unpopulatedInstancePerDevice.end(); ++it) {
        int deviceNum = it->first;
        unpopulatedInstancesPerDevice[deviceNum].push_back(it->second);
    }

    return copyFromAsyncImpl(populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);
}

void DistributedTensor::copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                                          map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                          Stream stream) {
    assert(!uninitialized());

    map<int, Event> populatedEventPerDevice;

    for (auto it = populatedInstancePerDevice.begin(); it != populatedInstancePerDevice.end(); ++it) {
        int deviceNum = it->first;
        populatedEventPerDevice[deviceNum] = stream.putEvent();
    }

    map<int, Event> deviceFinishedEvents =
        copyFromAsyncImpl(populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);

    for (auto entry : deviceFinishedEvents) {
        Event deviceFinishedEvent = entry.second;
        stream.waitEvent(deviceFinishedEvent);
    }
}

// Copies tensor instances when there is a device that should have one or more instance and has 0 instance,
// the function ensures that every device will have a tensor instance, so the remaining copying to do is same device
map<int, Event> DistributedTensor::copyFromAsyncImpl(map<int, Tensor> &populatedInstancePerDevice,
                                                     map<int, vector<Tensor>> &unpopulatedInstancesPerDevice,
                                                     map<int, Event> &populatedEventPerDevice) {
    assert(!uninitialized());

    assert(!populatedInstancePerDevice.empty());
    assert(!unpopulatedInstancesPerDevice.empty());

    if (unpopulatedInstancesPerDevice.empty())
        return map<int, Event>();

    set<int> devicesWithTensorInstance;

    for (auto it = populatedInstancePerDevice.begin(); it != populatedInstancePerDevice.end(); ++it) {
        int deviceNum = it->first;
        devicesWithTensorInstance.insert(deviceNum);
    }
    for (auto it = unpopulatedInstancesPerDevice.begin(); it != unpopulatedInstancesPerDevice.end(); ++it) {
        int deviceNum = it->first;
        devicesWithTensorInstance.insert(deviceNum);
    }

    // If any devices needs to be populated externally
    if (populatedInstancePerDevice.size() < devicesWithTensorInstance.size()) {
        // Adjacency groups are groups of gpu's, each adjacent to the next, that all need to have an instance populated
        // These groups' endpoints are a gpu that is adjacent to either 1. the first gpu, 2. the last gpu, 3. a gpu with
        // a populated instance
        vector<int> orderedGpus = MachineEvaluator::instance().getOrderedGpus();
        vector<deque<int>> adjacencyGroups;
        deque<int> currentGroup;
        for (int orderedGpuNum : orderedGpus) {
            // if GPU needs to be populated
            if (unpopulatedInstancesPerDevice.count(orderedGpuNum) == 1 && populatedInstancePerDevice.count(orderedGpuNum) == 0) {
                currentGroup.push_back(orderedGpuNum);
            } else if (populatedInstancePerDevice.count(orderedGpuNum) == 1) {
                if (!currentGroup.empty()) {
                    adjacencyGroups.push_back(currentGroup);
                    currentGroup.clear();
                }
            }
        }
        if (!currentGroup.empty())
            adjacencyGroups.push_back(currentGroup);

        // The only populated device is the CPU so copy to (up to) 2 well chosen gpus and this will form (up to) 3
        // adjaceny groups
        if (populatedInstancePerDevice.size() == 1 && populatedInstancePerDevice.count(MachineEvaluator::CPU_DEVICE_NUM) == 1) {
            assert(adjacencyGroups.size() == 1);
            currentGroup = adjacencyGroups.back();
            adjacencyGroups.clear();

            set<int> destinations;
            int orderIndex = currentGroup.size() / 4;
            destinations.insert(currentGroup[orderIndex]);
            orderIndex = 3 * currentGroup.size() / 4;
            if (destinations.count(currentGroup[orderIndex]) == 0)
                destinations.insert(currentGroup[orderIndex]);

            int copyFrom = MachineEvaluator::CPU_DEVICE_NUM;
            for (int copyTo : destinations)
                crossDeviceCopy(copyTo, copyFrom, populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);

            deque<int> newGroup;
            for (int gpuNum : currentGroup) {
                if (destinations.count(gpuNum) == 0) {
                    newGroup.push_back(gpuNum);
                } else {
                    if (!newGroup.empty()) {
                        adjacencyGroups.push_back(newGroup);
                        newGroup.clear();
                    }
                }
            }
        }

        // Now schedule cross-device copies starting from both ends of each group and working toward the middle.

        // Copy to CPU if needed
        if (unpopulatedInstancesPerDevice.count(MachineEvaluator::CPU_DEVICE_NUM) == 1 &&
            populatedInstancePerDevice.count(MachineEvaluator::CPU_DEVICE_NUM) == 0) {
            int copyFrom = populatedInstancePerDevice.begin()->first;
            crossDeviceCopy(MachineEvaluator::CPU_DEVICE_NUM,
                            copyFrom,
                            populatedInstancePerDevice,
                            unpopulatedInstancesPerDevice,
                            populatedEventPerDevice);
        }

        // If a group contains an endpoint, then copy from just one direction.
        // for(deque<int> currentGroup : adjacencyGroups) {
        for (unsigned int i = 0; i < adjacencyGroups.size(); ++i) {
            deque<int> currentGroup = adjacencyGroups[i];

            while (!currentGroup.empty()) {
                if (MachineEvaluator::instance().getAdjacentLowerGpu(currentGroup.front()) != MachineEvaluator::NONE) {
                    int copyFrom = MachineEvaluator::instance().getAdjacentLowerGpu(currentGroup.front());
                    int copyTo = currentGroup.front();
                    currentGroup.pop_front();

                    crossDeviceCopy(copyTo, copyFrom, populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);
                }

                if (currentGroup.empty())
                    break;

                if (MachineEvaluator::instance().getAdjacentHigherGpu(currentGroup.back()) != MachineEvaluator::NONE) {
                    int copyFrom = MachineEvaluator::instance().getAdjacentHigherGpu(currentGroup.back());
                    int copyTo = currentGroup.back();
                    currentGroup.pop_back();

                    crossDeviceCopy(copyTo, copyFrom, populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);
                }
            }
        }
    }

    // Now schedule internal copies to populate any remaining tensor instances.
    for (auto entry : unpopulatedInstancesPerDevice) {
        int deviceNum = entry.first;
        localDeviceCopy(deviceNum, populatedInstancePerDevice, unpopulatedInstancesPerDevice, populatedEventPerDevice);
    }

    return populatedEventPerDevice;
}

Tensor DistributedTensor::addInstance(TensorPlacement instancePlacement) {
    assert(!uninitialized());

    unique_lock<recursive_mutex> lck(*tensorMutex);

    Tensor instance = Tensor(instancePlacement, descriptor);
    (*instances)[instance.getTensorId()] = instance;

    return instance;
}

void DistributedTensor::removeInstance(unsigned long instanceId) {
    assert(!uninitialized());

    unique_lock<recursive_mutex> lck(*tensorMutex);

    assert(instances->count(instanceId) == 1);
    instances->erase(instanceId);
}

void DistributedTensor::removeInstance(TensorPlacement instancePlacement) {
    assert(!uninitialized());

    unique_lock<recursive_mutex> lck(*tensorMutex);

    assert(hasInstance(instancePlacement));
    Tensor instance = getInstance(instancePlacement);
    instances->erase(instance.getTensorId());
}

bool DistributedTensor::hasInstance(unsigned long instanceId) {
    assert(!uninitialized());

    return instances->count(instanceId) == 1;
}

bool DistributedTensor::hasInstance(TensorPlacement tensorPlacement) {
    assert(!uninitialized());

    for (auto it = instances->begin(); it != instances->end(); ++it) {
        if (it->second.getPlacement() == tensorPlacement)
            return true;
    }
    return false;
}

Tensor DistributedTensor::getInstance(unsigned long instanceId) {
    assert(!uninitialized());

    assert(instances->count(instanceId) == 1);
    return (*instances)[instanceId];
}

Tensor DistributedTensor::getInstance(TensorPlacement tensorPlacement) {
    assert(!uninitialized());

    for (auto it = instances->begin(); it != instances->end(); ++it) {
        if (it->second.getPlacement() == tensorPlacement)
            return it->second;
    }
    assert(false);
}

Tensor DistributedTensor::getNearestInstance(Tensor other) {
    assert(!uninitialized());

    assert(getNumInstances() > 0);
    if (getNumInstances() == 1)
        assert(getAnyInstance().getTensorId() != other.getTensorId());

    if (other.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        unsigned long nearestInstanceId = other.getTensorId();
        for (auto it = instances->begin(); it != instances->end(); ++it) {
            if (other.getTensorId() != it->first) {
                if (it->second.getPlacement() == TensorPlacement::MemDevices::CPU)
                    return it->second;
                else
                    nearestInstanceId = it->first;
            }
        }

        assert(nearestInstanceId != other.getTensorId());
        return getInstance(nearestInstanceId);

    } else if (other.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        int CPU_DISTANCE = 100;

        int otherGpuPciBusId = MachineEvaluator::instance().getGpuPciBusId(other.getPlacement().getDeviceNum());

        int smallestDistance = -1;
        unsigned long nearestInstanceId = other.getTensorId();
        for (auto it = instances->begin(); it != instances->end(); ++it) {
            unsigned long instanceTensorId = it->first;
            if (other.getTensorId() == instanceTensorId)
                continue;

            TensorPlacement instancePlacement = it->second.getPlacement();
            int instanceDistance =
                instancePlacement.getMemDevice() == TensorPlacement::MemDevices::CPU
                    ? CPU_DISTANCE
                    : abs(otherGpuPciBusId - MachineEvaluator::instance().getGpuPciBusId(instancePlacement.getDeviceNum()));
            if (instanceDistance < smallestDistance || smallestDistance == -1) {
                smallestDistance = instanceDistance;
                nearestInstanceId = instanceTensorId;
            }

            if (smallestDistance == 0)
                break;
        }

        // the assertions above should prevent the next line from ever triggering.
        assert(nearestInstanceId != other.getTensorId());
        return getInstance(nearestInstanceId);

    } else {
        assert(false);
    }
}

Tensor DistributedTensor::getNearestInstance(TensorPlacement tensorPlacement) {
    assert(!uninitialized());

    assert(getNumInstances() > 0);

    if (tensorPlacement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        if (hasInstance(TensorPlacement::MemDevices::CPU)) {
            return getInstance(TensorPlacement::MemDevices::CPU);
        } else {
            return getAnyInstance();
        }
    } else if (tensorPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        int CPU_DISTANCE = 100;

        int otherGpuPciBusId = MachineEvaluator::instance().getGpuPciBusId(tensorPlacement.getDeviceNum());

        auto it = instances->begin();
        TensorPlacement instancePlacement = it->second.getPlacement();
        int smallestDistance = instancePlacement.getMemDevice() == TensorPlacement::MemDevices::CPU
                                   ? CPU_DISTANCE
                                   : abs(otherGpuPciBusId - MachineEvaluator::instance().getGpuPciBusId(instancePlacement.getDeviceNum()));
        unsigned long nearestInstanceId = it->first;
        for (; it != instances->end(); ++it) {
            int instanceDistance =
                instancePlacement.getMemDevice() == TensorPlacement::MemDevices::CPU
                    ? CPU_DISTANCE
                    : abs(otherGpuPciBusId - MachineEvaluator::instance().getGpuPciBusId(instancePlacement.getDeviceNum()));
            if (instanceDistance < smallestDistance) {
                smallestDistance = instanceDistance;
                nearestInstanceId = it->first;
            }
            if (smallestDistance == 0)
                break;
        }

        return getInstance(nearestInstanceId);
    } else {
        assert(false);
    }
}

unordered_map<unsigned long, Tensor> DistributedTensor::getInstances() {
    assert(!uninitialized());
    return *instances;
}
