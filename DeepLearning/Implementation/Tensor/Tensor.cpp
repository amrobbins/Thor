#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace ThorImplementation;
using namespace std;

static CuFileInitializer cufile_global_initializer;

atomic<unsigned long> Tensor::nextInstanceId(1);

uint32_t murmurHash(const void *key, int32_t len, uint32_t seed) {
    const uint32_t m = 0x5bd1e995;
    const int32_t r = 24;
    uint32_t h = seed ^ len;
    const uint8_t *data = (const uint8_t *)key;

    while (len >= 4) {
        uint32_t k = *(uint32_t *)data;
        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;
        data += 4;
        len -= 4;
    }

    switch (len) {
        case 3:
            h ^= data[2] << 16;
            // fallthrough
        case 2:
            h ^= data[1] << 8;
            // fallthrough
        case 1:
            h ^= data[0];
            h *= m;
            // fallthrough
    }

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

uint64_t murmurHash64(const void *key, int32_t len, uint64_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int32_t r = 47;

    uint64_t h = seed ^ (len * m);

    const uint8_t *data = (const uint8_t *)key;

    while (len >= 8) {
        uint64_t k = *(uint64_t *)data;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
        data += 8;
        len -= 8;
    }

    switch (len) {
        case 7:
            h ^= uint64_t(data[6]) << 48;
            // fallthrough
        case 6:
            h ^= uint64_t(data[5]) << 40;
            // fallthrough
        case 5:
            h ^= uint64_t(data[4]) << 32;
            // fallthrough
        case 4:
            h ^= uint64_t(data[3]) << 24;
            // fallthrough
        case 3:
            h ^= uint64_t(data[2]) << 16;
            // fallthrough
        case 2:
            h ^= uint64_t(data[1]) << 8;
            // fallthrough
        case 1:
            h ^= uint64_t(data[0]);
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

uint32_t Tensor::getThreadIdHash(uint32_t seed, bool reproducible) {
    uint64_t long_hash = getThreadIdHash64(seed, reproducible);
    return long_hash ^ (long_hash >> 31);
}

static inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t Tensor::getThreadIdHash64(uint64_t seed, bool reproducible) {
    uint64_t threadId = omp_get_thread_num();
    uint64_t tag = threadId;
    if (!reproducible) {
        thread_local int tl;
        tag ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&tl));
    }
    return splitmix64(seed) ^ splitmix64(tag);
}

Tensor::Tensor() : ReferenceCounted() {}

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
    // All tensors end on a 32 byte boundary so that kernels can overshoot when writing arrays in chunks of the maximum sized type
    // (double4) without risking accessing another memory block. Consider the overhead of having 1 million tensors instantiated: the
    // overhead is less than 32 MB of GPU memory.
    memBytes = (memBytes + 31) / 32;
    memBytes *= 32;

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

void Tensor::attachFile(const std::string &fileName, const off_t fileOffset, const FileAccess fileAccessRequirement, bool createEmptyFile) {
    assert(getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU ||
           getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    if (!this->fileName.empty())
        detachFile();
    assert(this->fileName.empty());
    this->fileName = fileName;

    this->fileAccessRequirement = fileAccessRequirement;

    int32_t o_flags = O_CLOEXEC;
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        o_flags |= O_DIRECT;
    if (fileAccessRequirement == FileAccess::READ_ONLY)
        o_flags |= O_RDONLY;
    else if (fileAccessRequirement == FileAccess::WRITE_ONLY)
        o_flags |= O_WRONLY;
    else
        o_flags |= O_RDWR;
    if (createEmptyFile && fileAccessRequirement != FileAccess::READ_ONLY)
        o_flags |= O_CREAT | O_TRUNC;

    // Get a handle to the file with the necessary flags
    if (o_flags & O_CREAT) {
        fileDescriptor = open(fileName.c_str(), o_flags, 0644);
    } else {
        fileDescriptor = open(fileName.c_str(), o_flags);
    }
    if (fileDescriptor < 0) {
        printf("Error opening file %s  fileDescriptor %d errno %d\n", fileName.c_str(), fileDescriptor, errno);
        printf("open(\"%s\", flags=0x%x) failed: %s (errno=%d), createEmptyFile=%d, access=%d\n",
               fileName.c_str(),
               o_flags,
               strerror(errno),
               errno,
               (int)createEmptyFile,
               (int)fileAccessRequirement);
        fflush(stdout);
        assert(fileDescriptor >= 0);
    }
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        // Note: Its critical to zero-initialize cuFileDescriptor, which is done with the {}
        CUfileDescr_t cuFileDescriptor{};
        CUfileError_t cuFileError;
        cuFileDescriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cuFileDescriptor.handle.fd = fileDescriptor;
        cuFileError = cuFileHandleRegister(&gpuDirectStorageCuFileHandle, &cuFileDescriptor);
        assert(cuFileError.err == CU_FILE_SUCCESS);
    }

    // GpuDirect Storage takes pointers to its parameters:
    this->gpuDirectStorageFileOffset = fileOffset;
    this->gpuDirectStorageSize = getArraySizeInBytes();
}

void Tensor::detachFile() {
    if (fileName.empty()) {
        return;
    }

    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        cuFileHandleDeregister(gpuDirectStorageCuFileHandle);
    }

    close(fileDescriptor);

    fileName.clear();
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
    detachFile();  // detachFile is a NOP when there is no attached file

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
        ScopedGpu scopedGpu(getPlacement().getDeviceNum());

        cudaError_t cudaStatus = cudaFree(mem);
        assert(cudaStatus == cudaSuccess);
    } else {
        assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
    mem = nullptr;
}

template <typename ElementDataType>
ElementDataType *Tensor::getMemPtr() {
    assert(isInitialized());
    assert(mem != nullptr);
    // Ensure that if the convenience template parameter ElementDataType is used that it agrees with the descriptor
    if (!(is_same<ElementDataType, void>::value)) {
        if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
            assert((is_same<ElementDataType, half>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
            assert((is_same<ElementDataType, float>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
            assert((is_same<ElementDataType, int8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
            assert((is_same<ElementDataType, int16_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
            assert((is_same<ElementDataType, int32_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
            assert((is_same<ElementDataType, uint8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
            assert((is_same<ElementDataType, uint16_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
            assert((is_same<ElementDataType, uint32_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
            assert((is_same<ElementDataType, bool>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
            assert((is_same<ElementDataType, uint8_t>::value));
        else
            assert(false);
    }
    return (ElementDataType *)mem;
}

template <typename ElementDataType>
ElementDataType Tensor::getElement(vector<unsigned long> dimensionIndex) {
    assert(!uninitialized());

#ifdef THOR_DEBUG
    // This seems like too much overhead to get just one element, so it is explicitly removed for release.
    if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
        assert((is_same<ElementDataType, half>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
        assert((is_same<ElementDataType, float>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
        assert((is_same<ElementDataType, int8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
        assert((is_same<ElementDataType, int16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
        assert((is_same<ElementDataType, int32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
        assert((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
        assert((is_same<ElementDataType, uint16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
        assert((is_same<ElementDataType, uint32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
        assert((is_same<ElementDataType, bool>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
        assert((is_same<ElementDataType, uint8_t>::value));
    else
        assert(false);
#endif

    assert(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    return *((ElementDataType *)getDescriptor().getChunkAddress(dimensionIndex, mem));
}

template <typename ElementDataType>
void Tensor::setElement(std::vector<unsigned long> dimensionIndex, const ElementDataType &value) {
    assert(!uninitialized());

#ifdef THOR_DEBUG
    // This seems like too much overhead to get just one element, so it is explicitly removed for release.
    if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
        assert((is_same<ElementDataType, half>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
        assert((is_same<ElementDataType, float>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
        assert((is_same<ElementDataType, int8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
        assert((is_same<ElementDataType, int16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
        assert((is_same<ElementDataType, int32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
        assert((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
        assert((is_same<ElementDataType, uint16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
        assert((is_same<ElementDataType, uint32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
        assert((is_same<ElementDataType, bool>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
        assert((is_same<ElementDataType, uint8_t>::value));
    else
        assert(false);
#endif

    assert(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    *((ElementDataType *)getDescriptor().getChunkAddress(dimensionIndex, mem)) = value;
}

template <typename ElementDataType>
ElementDataType *Tensor::getElementPointer(std::vector<unsigned long> dimensionIndex) {
    assert(!uninitialized());

#ifdef THOR_DEBUG
    // This seems like too much overhead to get just one element, so it is explicitly removed for release.
    if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
        assert((is_same<ElementDataType, half>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
        assert((is_same<ElementDataType, float>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
        assert((is_same<ElementDataType, int8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
        assert((is_same<ElementDataType, int16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
        assert((is_same<ElementDataType, int32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
        assert((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
        assert((is_same<ElementDataType, uint16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
        assert((is_same<ElementDataType, uint32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
        assert((is_same<ElementDataType, bool>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
        assert((is_same<ElementDataType, uint8_t>::value));
    else
        assert(false);
#endif

    assert(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    return (ElementDataType *)getDescriptor().getChunkAddress(dimensionIndex, mem);
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
    assert(getPlacement() == TensorPlacement::MemDevices::CPU || getPlacement() == TensorPlacement::MemDevices::GPU);
    assert(!fileName.empty());
    assert(fileAccessRequirement != FileAccess::READ_ONLY);

    vector<int> devicesInvolved;
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(source.getPlacement().getDeviceNum());
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(getPlacement().getDeviceNum());
    if (!devicesInvolved.empty())
        assert(stream.getGpuNum() == devicesInvolved[0] || (devicesInvolved.size() == 2 && stream.getGpuNum() == devicesInvolved[1]));
    copyFromAsync(source, stream, false);
}

struct CheckIoBytesParams : HostFunctionArgsBase {
    CheckIoBytesParams(const ssize_t expectedBytes, const ssize_t *actualBytes) : expectedBytes(expectedBytes), actualBytes(actualBytes) {}
    const ssize_t expectedBytes;
    const ssize_t *actualBytes;
};

void checkBytesIoOp(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    assert(baseParams != nullptr);
    CheckIoBytesParams *checkIoBytesParams = dynamic_cast<CheckIoBytesParams *>(baseParams);
    assert(checkIoBytesParams != nullptr);

    if (*(checkIoBytesParams->actualBytes) != checkIoBytesParams->expectedBytes) {
        string errorString;
        errorString = "GpuDirect Storage transfer initiated to transfer " + to_string(checkIoBytesParams->expectedBytes) +
                      " bytes, but gpuDirectStorage reported that it transferred " + to_string(*(checkIoBytesParams->actualBytes)) +
                      " bytes.";
        // FIXME: This is thrown by a worker thread. Does it need to put the exception into a sync queue for the main thread?
        throw runtime_error(errorString.c_str());
    }
}

struct PerformReadParams : HostFunctionArgsBase {
    PerformReadParams(void *memPtr, size_t totalBytesToRead, const string &fileName, const off_t fileOffset, const int32_t fileDescriptor)
        : memPtr(memPtr), totalBytesToRead(totalBytesToRead), fileName(fileName), fileOffset(fileOffset), fileDescriptor(fileDescriptor) {}
    const void *memPtr;
    const size_t totalBytesToRead;
    const string fileName;
    const off_t fileOffset;
    const int32_t fileDescriptor;
};

void performRead(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    assert(baseParams != nullptr);
    PerformReadParams *performReadParams = dynamic_cast<PerformReadParams *>(baseParams);
    assert(performReadParams != nullptr);

    const void *memPtr = performReadParams->memPtr;
    const size_t totalBytesToRead = performReadParams->totalBytesToRead;
    const string fileName = performReadParams->fileName;
    const off_t fileOffset = performReadParams->fileOffset;
    const int32_t fileDescriptor = performReadParams->fileDescriptor;

    ssize_t bytesRead = 0;
    size_t bytesLeftToRead = totalBytesToRead;
    if (lseek(fileDescriptor, fileOffset, SEEK_SET) < 0) {
        string errorString = "Error seeking to " + to_string(fileOffset) + " in file " + fileName;
        throw runtime_error(errorString);
    }

    uint8_t *runningMemPtr = (uint8_t *)memPtr;
    while (bytesLeftToRead > 0) {
        bytesRead = read(fileDescriptor, runningMemPtr, bytesLeftToRead);

        if (bytesRead <= 0 || (size_t)bytesRead > bytesLeftToRead) {
            close(fileDescriptor);
            string errorString = "read failed for file " + fileName + ", requesting " + to_string(bytesLeftToRead) + " bytes at offset " +
                                 to_string(totalBytesToRead - bytesLeftToRead) + ".  bytesRead value " + to_string(bytesRead);
            throw runtime_error(errorString);
        }

        bytesLeftToRead -= bytesRead;
        runningMemPtr += bytesRead;
    }
}

struct PerformWriteParams : HostFunctionArgsBase {
    PerformWriteParams(void *memPtr, size_t totalBytesToWrite, const string &fileName, const off_t fileOffset, const int32_t fileDescriptor)
        : memPtr(memPtr),
          totalBytesToWrite(totalBytesToWrite),
          fileName(fileName),
          fileOffset(fileOffset),
          fileDescriptor(fileDescriptor) {}
    const void *memPtr;
    const size_t totalBytesToWrite;
    const string fileName;
    const off_t fileOffset;
    const int32_t fileDescriptor;
};

void performWrite(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    assert(baseParams != nullptr);
    PerformWriteParams *performWriteParams = dynamic_cast<PerformWriteParams *>(baseParams);
    assert(performWriteParams != nullptr);

    const void *memPtr = performWriteParams->memPtr;
    const size_t totalBytesToWrite = performWriteParams->totalBytesToWrite;
    const string fileName = performWriteParams->fileName;
    const off_t fileOffset = performWriteParams->fileOffset;
    const int32_t fileDescriptor = performWriteParams->fileDescriptor;

    ssize_t bytesWritten = 0;
    size_t bytesLeftToWrite = totalBytesToWrite;
    if (lseek(fileDescriptor, fileOffset, SEEK_SET) < 0) {
        string errorString = "Error seeking to " + to_string(fileOffset) + " in file " + fileName;
        throw runtime_error(errorString);
    }

    uint8_t *runningMemPtr = (uint8_t *)memPtr;
    while (bytesLeftToWrite > 0) {
        bytesWritten = write(fileDescriptor, runningMemPtr, bytesLeftToWrite);

        if (bytesWritten <= 0 || (size_t)bytesWritten > bytesLeftToWrite) {
            close(fileDescriptor);
            string errorString = "write failed for file " + fileName + ", requesting " + to_string(bytesLeftToWrite) + " bytes at offset " +
                                 to_string(totalBytesToWrite - bytesLeftToWrite) + ".  bytesWritten value " + to_string(bytesWritten);
            throw runtime_error(errorString);
        }

        bytesLeftToWrite -= bytesWritten;
        runningMemPtr += bytesWritten;
    }
}

void Tensor::loadFromFile(Stream stream) {
    assert(!uninitialized());
    assert(getPlacement() == TensorPlacement::MemDevices::CPU || getPlacement() == TensorPlacement::MemDevices::GPU);
    assert(!fileName.empty());
    assert(fileAccessRequirement != FileAccess::WRITE_ONLY);

    if (getPlacement() == TensorPlacement::MemDevices::GPU) {
        gpuDirectStorageBytesAccessed = 0;
        CUfileError_t cuFileError;
        cuFileError = cuFileReadAsync(gpuDirectStorageCuFileHandle,
                                      getMemPtr(),
                                      &gpuDirectStorageSize,
                                      &gpuDirectStorageFileOffset,
                                      &gpuDirectStorageBufOffset,
                                      &gpuDirectStorageBytesAccessed,
                                      stream);
        assert(cuFileError.err == CU_FILE_SUCCESS);
        std::unique_ptr<HostFunctionArgsBase> args(new CheckIoBytesParams(getArraySizeInBytes(), &gpuDirectStorageBytesAccessed));
        stream.enqueueHostFunction(checkBytesIoOp, std::move(args));
    } else {
        std::unique_ptr<HostFunctionArgsBase> args(
            new PerformReadParams(getMemPtr(), getArraySizeInBytes(), fileName, gpuDirectStorageFileOffset, fileDescriptor));
        stream.enqueueHostFunction(performRead, std::move(args));
    }
}

void Tensor::dumpToFile(Stream stream) {
    assert(!uninitialized());
    assert(getPlacement() == TensorPlacement::MemDevices::CPU || getPlacement() == TensorPlacement::MemDevices::GPU);
    assert(!fileName.empty());
    assert(fileAccessRequirement != FileAccess::READ_ONLY);

    if (getPlacement() == TensorPlacement::MemDevices::GPU) {
        gpuDirectStorageBytesAccessed = 0;
        CUfileError_t cuFileError;
        cuFileError = cuFileWriteAsync(gpuDirectStorageCuFileHandle,
                                       getMemPtr(),
                                       &gpuDirectStorageSize,
                                       &gpuDirectStorageFileOffset,
                                       &gpuDirectStorageBufOffset,
                                       &gpuDirectStorageBytesAccessed,
                                       stream);
        assert(cuFileError.err == CU_FILE_SUCCESS);
        std::unique_ptr<HostFunctionArgsBase> args(new CheckIoBytesParams(getArraySizeInBytes(), &gpuDirectStorageBytesAccessed));
        stream.enqueueHostFunction(checkBytesIoOp, std::move(args));
    } else {
        std::unique_ptr<HostFunctionArgsBase> args(
            new PerformWriteParams(getMemPtr(), getArraySizeInBytes(), fileName, gpuDirectStorageFileOffset, fileDescriptor));
        stream.enqueueHostFunction(performWrite, std::move(args));
    }
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

void Tensor::copyFromAsync(Tensor source, Stream copyStream, bool mustPreserveSourceValue) {
    assert(!uninitialized());
    assert(!source.uninitialized());

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
    // If the destination data type is smaller than the source data type, then this is only supported when mustPreserveSourceValue is false.
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
    // On GPU this would require device synchronization so this is not supported.
    assert(placement.getMemDevice() != TensorPlacement::MemDevices::GPU);
    assert(placement.getMemDevice() == TensorPlacement::MemDevices::CPU);

    uint64_t numBytes;
    if (numElements == 0) {
        numBytes = getArraySizeInBytes();
    } else {
        if (getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            assert(numElements % 8 == 0);
            numBytes = numElements / 8;
        } else {
            numBytes = numElements * (getArraySizeInBytes() / getTotalNumElements());
        }
    }

    // invoke global memset instead of member function memset
    ::memset(mem, value, numBytes);
}

struct IdentityMatrixArgs : HostFunctionArgsBase {
    IdentityMatrixArgs(Tensor tensor) : tensor(tensor) {}
    Tensor tensor;
};

void fillCpuIdentityMatrixOnes(void *data) {
    HostFunctionArgsBase *baseArgs = static_cast<HostFunctionArgsBase *>(data);
    assert(baseArgs != nullptr);
    IdentityMatrixArgs *args = dynamic_cast<IdentityMatrixArgs *>(baseArgs);
    assert(args != nullptr);

    assert(args->tensor.getPlacement() == TensorPlacement::MemDevices::CPU);
    TensorDescriptor::DataType dataType = args->tensor.getDataType();

    uint64_t N = args->tensor.getDimensions()[0];
    if (dataType == TensorDescriptor::DataType::FP16) {
        half *mem = args->tensor.getMemPtr<half>();
        for (uint32_t i = 0; i < N; ++i)
            mem[i * N + i] = 1.0f;
    } else {
        float *mem = args->tensor.getMemPtr<float>();
        for (uint32_t i = 0; i < N; ++i)
            mem[i * N + i] = 1.0f;
    }
}

Tensor Tensor::identityMatrix(uint32_t N, TensorPlacement placement, TensorDescriptor::DataType dataType, Stream stream) {
    assert(dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32);
    Tensor tensor(placement, TensorDescriptor(dataType, {N, N}));

    if (placement == TensorPlacement::MemDevices::CPU) {
        tensor.clearAsync(stream);
        std::unique_ptr<HostFunctionArgsBase> args(new IdentityMatrixArgs(tensor));
        stream.enqueueHostFunction(fillCpuIdentityMatrixOnes, std::move(args));
    } else {
        tensor.clearAsync(stream);
        tensor.fillGpuIdentityMatrixOnes(stream);
    }

    return tensor;
}

Tensor Tensor::zeros(TensorPlacement placement, TensorDescriptor descriptor, Stream stream) {
    Tensor tensor(placement, descriptor);
    tensor.fillZero(stream);
    return tensor;
}

Tensor Tensor::randoms(TensorPlacement placement, TensorDescriptor descriptor, Stream stream, double minValue, double maxValue) {
    Tensor tensor(placement, descriptor);
    tensor.fillRandom(minValue, maxValue, stream);
    return tensor;
}

Tensor Tensor::values(TensorPlacement placement, TensorDescriptor descriptor, Stream stream, double value) {
    Tensor tensor(placement, descriptor);
    tensor.fill(value, stream);
    return tensor;
}

struct MemsetArgs : HostFunctionArgsBase {
    MemsetArgs(Tensor tensor, int8_t value, uint64_t numElements) : tensor(tensor), value(value), numElements(numElements) {}
    Tensor tensor;
    int8_t value;
    uint64_t numElements;
};

void callMemsetOnTensor(void *data) {
    HostFunctionArgsBase *baseArgs = static_cast<HostFunctionArgsBase *>(data);
    assert(baseArgs != nullptr);
    MemsetArgs *args = dynamic_cast<MemsetArgs *>(baseArgs);
    assert(args != nullptr);
    args->tensor.memset(args->value, args->numElements);
}

void Tensor::memsetAsync(Stream stream, int8_t value, uint64_t numElements) {
    if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
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
    } else {
        std::unique_ptr<HostFunctionArgsBase> args(new MemsetArgs(*this, value, numElements));
        stream.enqueueHostFunction(callMemsetOnTensor, std::move(args));
    }
}

void Tensor::clear() { this->memset(0); }

void Tensor::clearAsync(Stream stream) { this->memsetAsync(stream, 0); }

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

struct FillRandomArgs : HostFunctionArgsBase {
    FillRandomArgs(Tensor tensor, double minValue, double maxValue) : tensor(tensor), minValue(minValue), maxValue(maxValue) {}
    Tensor tensor;
    double minValue;
    double maxValue;
};

void fillCpuRandom(void *data) {
    HostFunctionArgsBase *baseArgs = static_cast<HostFunctionArgsBase *>(data);
    assert(baseArgs != nullptr);
    FillRandomArgs *args = dynamic_cast<FillRandomArgs *>(baseArgs);
    assert(args != nullptr);

    assert(args->tensor.getPlacement() == TensorPlacement::MemDevices::CPU);
    TensorDescriptor::DataType dataType = args->tensor.getDataType();

    Tensor tensor = args->tensor;
    double minValue = args->minValue;
    double maxValue = args->maxValue;
    if (minValue > maxValue)
        swap(minValue, maxValue);
    uint64_t numElements = tensor.getTotalNumElements();
    const uint32_t numProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
    const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;

    if (dataType == TensorDescriptor::DataType::FP16) {
        half *mem = tensor.getMemPtr<half>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_real_distribution<double> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (half)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_real_distribution<double> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (half)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        float *mem = tensor.getMemPtr<float>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_real_distribution<double> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (float)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_real_distribution<double> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (float)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        int8_t *mem = tensor.getMemPtr<int8_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<int16_t> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (int8_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<int16_t> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (int8_t)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        int16_t *mem = tensor.getMemPtr<int16_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<int32_t> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (int16_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<int32_t> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (int16_t)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        int32_t *mem = tensor.getMemPtr<int32_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<int64_t> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (int32_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<int64_t> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (int32_t)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        uint8_t *mem = tensor.getMemPtr<uint8_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<uint16_t> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (uint8_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<uint16_t> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (uint8_t)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        uint16_t *mem = tensor.getMemPtr<uint16_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<uint32_t> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (uint16_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<uint32_t> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (uint16_t)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        uint32_t *mem = tensor.getMemPtr<uint32_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<uint64_t> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (uint32_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<uint64_t> dis(minValue, maxValue);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (uint32_t)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::BOOLEAN) {
        bool *mem = tensor.getMemPtr<bool>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<uint8_t> dis(0, 2);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (bool)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<uint8_t> dis(0, 2);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (bool)dis(gen);
            }
        }
    } else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
        numElements = (numElements + 7) / 8;
        const uint32_t numProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
        const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;

        uint8_t *mem = tensor.getMemPtr<uint8_t>();
        if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
            {
                random_device rd;
                uint32_t seed = Tensor::getThreadIdHash(rd());
                mt19937 gen(seed);
                uniform_int_distribution<uint16_t> dis(0, 256);

#pragma omp for schedule(static, elementsPerThread)
                for (uint64_t i = 0; i < numElements; ++i) {
                    mem[i] = (uint8_t)dis(gen);
                }
            }
        } else {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<uint16_t> dis(0, 256);

            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = (uint8_t)dis(gen);
            }
        }
    }
}

void Tensor::fillRandom(double minValue, double maxValue, Stream stream) {
    if (maxValue < minValue)
        swap(maxValue, minValue);

    if (getPlacement() == TensorPlacement::MemDevices::CPU) {
        std::unique_ptr<HostFunctionArgsBase> args(new FillRandomArgs(*this, minValue, maxValue));
        stream.enqueueHostFunction(fillCpuRandom, std::move(args));
    } else {
        TensorDescriptor::DataType dataType = getDataType();
        if (dataType == TensorDescriptor::DataType::FP16) {
            launchGpuFillRandom<half>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::FP32) {
            launchGpuFillRandom<float>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT8) {
            // Since for int I am rounding down for integer types, adding just under 1.0 to the continuous distribution gives
            // the range [minValue, maxValue] where minValue and maxValue have equal likelihood of being generated to all other values.
            // Note that converting to an integer truncates and so rounds toward 0.
            if (maxValue > 0) {
                // integer rounding (truncation) rounds away from maxValue in this case
                if (maxValue == int32_t(maxValue))
                    maxValue += 0.99999;
            }
            if (minValue < 0) {
                // integer rounding (truncation) rounds away from minValue in this case
                if (minValue == int32_t(minValue))
                    minValue -= 0.99999;
            }
            launchGpuFillRandom<int8_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT16) {
            if (maxValue > 0) {
                // integer rounding (truncation) rounds away from maxValue in this case
                if (maxValue == int32_t(maxValue))
                    maxValue += 0.99999;
            }
            if (minValue < 0) {
                // integer rounding (truncation) rounds away from minValue in this case
                if (minValue == int32_t(minValue))
                    minValue -= 0.99999;
            }
            launchGpuFillRandom<int16_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT32) {
            if (maxValue > 0) {
                // integer rounding (truncation) rounds away from maxValue in this case
                if (maxValue == int32_t(maxValue))
                    maxValue += 0.99999;
            }
            if (minValue < 0) {
                // integer rounding (truncation) rounds away from minValue in this case
                if (minValue == int32_t(minValue))
                    minValue -= 0.99999;
            }
            launchGpuFillRandom<int32_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT8) {
            if (maxValue == uint32_t(maxValue))
                maxValue += 0.99999;
            launchGpuFillRandom<uint8_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT16) {
            if (maxValue == uint32_t(maxValue))
                maxValue += 0.99999;
            launchGpuFillRandom<uint16_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT32) {
            if (maxValue == uint32_t(maxValue))
                maxValue += 0.99999;
            launchGpuFillRandom<uint32_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::BOOLEAN) {
            launchGpuFillRandom<bool>(getMemPtr(), getTotalNumElements(), 0, 1.999999, stream);
        } else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            launchGpuFillRandom<uint8_t>(getMemPtr(), (getTotalNumElements() + 7) / 8, 0, 255.999999, stream);
        }
    }
}

void Tensor::fillZero(Stream dataStream) { this->fill(0.0, dataStream); }

struct CpuFillParams : HostFunctionArgsBase {
    CpuFillParams(double value, Tensor tensor) : value(value), tensor(tensor) {}

    double value;
    Tensor tensor;
};

void fillValue(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    assert(baseParams != nullptr);
    CpuFillParams *cpuFillParams = dynamic_cast<CpuFillParams *>(baseParams);
    assert(cpuFillParams != nullptr);

    uint64_t numElements = cpuFillParams->tensor.getTotalNumElements();
    const uint32_t numProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
    const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;

    TensorDescriptor::DataType dataType = cpuFillParams->tensor.getDataType();

    if (dataType == TensorDescriptor::DataType::FP16) {
        half *mem = cpuFillParams->tensor.getMemPtr<half>();
        half value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::FP32) {
        float *mem = cpuFillParams->tensor.getMemPtr<float>();
        float value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::INT8) {
        int8_t *mem = cpuFillParams->tensor.getMemPtr<int8_t>();
        int8_t value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::INT16) {
        int16_t *mem = cpuFillParams->tensor.getMemPtr<int16_t>();
        int16_t value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::INT32) {
        int32_t *mem = cpuFillParams->tensor.getMemPtr<int32_t>();
        int32_t value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::UINT8) {
        uint8_t *mem = cpuFillParams->tensor.getMemPtr<uint8_t>();
        uint8_t value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::UINT16) {
        uint16_t *mem = cpuFillParams->tensor.getMemPtr<uint16_t>();
        uint16_t value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::UINT32) {
        uint32_t *mem = cpuFillParams->tensor.getMemPtr<uint32_t>();
        uint32_t value = cpuFillParams->value;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::BOOLEAN) {
        bool *mem = cpuFillParams->tensor.getMemPtr<bool>();
        bool value = cpuFillParams->value ? true : false;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
        uint8_t *mem = cpuFillParams->tensor.getMemPtr<uint8_t>();
        uint8_t value = cpuFillParams->value;
        if (value)
            value = 0b11111111;
        numElements = (numElements + 7) / 8;
        if (numProcs > 1) {
#pragma omp parallel for schedule(static, elementsPerThread) shared(mem, value, elementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    }
}

// Note: value will be cast to the type of the tensor elements
void Tensor::fill(double value, Stream stream) {
    TensorDescriptor::DataType dataType = getDataType();
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        std::unique_ptr<HostFunctionArgsBase> args(new CpuFillParams(value, *this));
        stream.enqueueHostFunction(fillValue, std::move(args));
    } else {
        if (dataType == TensorDescriptor::DataType::FP16) {
            launchFillValueGpuKernel<half>(
                (half)(float)value, (half *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::FP32) {
            launchFillValueGpuKernel<float>(value, (float *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT8) {
            launchFillValueGpuKernel<uint8_t>(value, (uint8_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT16) {
            launchFillValueGpuKernel<uint16_t>(
                value, (uint16_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT32) {
            launchFillValueGpuKernel<uint32_t>(
                value, (uint32_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT8) {
            launchFillValueGpuKernel<int8_t>(value, (int8_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT16) {
            launchFillValueGpuKernel<int16_t>(value, (int16_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT32) {
            launchFillValueGpuKernel<int32_t>(value, (int32_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::BOOLEAN) {
            launchFillValueGpuKernel<bool>(value, (bool *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            uint8_t packedValue = 0;
            if (value)
                packedValue = 0b11111111;
            uint64_t numPackedBytes = (getTotalNumElements() + 7) / 8;
            launchFillValueGpuKernel<uint8_t>(packedValue, (uint8_t *)getMemPtr(), numPackedBytes, getPlacement().getDeviceNum(), stream);
        } else {
            assert(false);
        }
    }
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

template void *Tensor::getMemPtr();
template half *Tensor::getMemPtr();
template float *Tensor::getMemPtr();
template int8_t *Tensor::getMemPtr();
template int16_t *Tensor::getMemPtr();
template int32_t *Tensor::getMemPtr();
template uint8_t *Tensor::getMemPtr();
template uint16_t *Tensor::getMemPtr();
template uint32_t *Tensor::getMemPtr();

template half Tensor::getElement(vector<unsigned long> dimensionIndex);
template float Tensor::getElement(vector<unsigned long> dimensionIndex);
template int8_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template int16_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template int32_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint8_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint16_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint32_t Tensor::getElement(vector<unsigned long> dimensionIndex);

template void Tensor::setElement(vector<unsigned long> dimensionIndex, const half &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const float &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int8_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int16_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int32_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint8_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint16_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint32_t &value);

template void *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template half *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template float *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int8_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int16_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int32_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint8_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint16_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint32_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
