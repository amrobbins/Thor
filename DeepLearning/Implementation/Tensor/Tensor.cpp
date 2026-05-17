#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include <optional>
#include <cmath>
#include <limits>

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;
using namespace std;

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

uint32_t Tensor::getThreadIdHash(uint32_t seed) {
    uint64_t long_hash = getThreadIdHash64(seed);
    return long_hash ^ (long_hash >> 31);
}

static inline uint64_t splitmixA64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t splitmixB64(uint64_t x) noexcept {
    x += 0xda942042e4dd58b5ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t Tensor::getThreadIdHash64(uint64_t seed) {
    uint64_t threadId = omp_get_thread_num();
    uint64_t tag = threadId;
    return splitmixA64(seed) ^ splitmixB64(tag);
}

Tensor::Tensor() : ReferenceCounted() {}

Tensor::Tensor(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes) {
    construct(placement, descriptor, alignmentBytes);
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
    THOR_THROW_IF_FALSE(!uninitialized());
    return instanceId == other.instanceId;
}

bool Tensor::operator!=(const Tensor &other) const {
    THOR_THROW_IF_FALSE(!uninitialized());
    return instanceId != other.instanceId;
}

bool Tensor::operator<(const Tensor &other) const {
    THOR_THROW_IF_FALSE(!uninitialized());
    return instanceId < other.instanceId;
}

void Tensor::construct(TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignmentBytes) {
    ReferenceCounted::initialize();

    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
                        placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(placement.getDeviceNum() >= 0);
    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU)
        THOR_THROW_IF_FALSE(placement.getDeviceNum() == 0);
    else
        THOR_THROW_IF_FALSE(placement.getDeviceNum() < (int)MachineEvaluator::instance().getNumGpus());
    this->placement = placement;

    this->descriptor = descriptor;
    storageElementOffset = 0;
    storageNumElements = descriptor.getTotalNumElements();
    customStridesElements.clear();
    descriptorOverridden = false;

    instanceId = nextInstanceId.fetch_add(1);

    allocateMemory(alignmentBytes);
}

static inline bool isPow2(std::size_t x) { return x && ((x & (x - 1)) == 0); }

void Tensor::allocateMemory(uint32_t alignmentBytes) {
    cudaError_t cudaStatus;

    unsigned long numElements = descriptor.getTotalNumElements();
    THOR_THROW_IF_FALSE(numElements > 0);

    if (alignmentBytes != 0)
        THOR_THROW_IF_FALSE(isPow2(alignmentBytes));

    unsigned long memBytes;
    memBytes = descriptor.getArraySizeInBytes();
    // All tensors add 32 bytes of padding to make writing up to a full double4 past the end safe.
    memBytes += 32;

    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        if (alignmentBytes <= 256) {
            // Cuda gives this natively
            cudaStatus = cudaHostAlloc(&mem, memBytes, cudaHostAllocPortable);
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        } else {
            void *p = nullptr;
            int prc = posix_memalign(&p, alignmentBytes, memBytes);
            if (prc != 0 || !p) {
                throw std::runtime_error(std::string("posix_memalign failed: ") + std::strerror(prc));
            }
            // Pin it so it still behaves like cudaHostAlloc memory for GPU transfers.
            cudaStatus = cudaHostRegister(p, memBytes, cudaHostRegisterPortable);
            if (cudaStatus != cudaSuccess) {
                free(p);
                throw std::runtime_error(std::string("cudaHostRegister failed: ") + cudaGetErrorString(cudaStatus));
            }
            mem = p;
            cpuMemPinnedViaCudaHostRegister = true;
        }
    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(placement.getDeviceNum());
        cudaStatus = cudaMalloc(&mem, memBytes);
        if (cudaStatus != cudaSuccess) {
            printf("cudaStatus %d\n", cudaStatus);
            printf("%s\n", cudaGetErrorString(cudaStatus));
            fflush(stdout);
        }
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
    } else {
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
                            placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
}

void Tensor::attachFile(const std::string &fileName, const off_t fileOffset, const FileAccess fileAccessRequirement, bool createEmptyFile) {
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU ||
                        getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    if (!this->fileName.empty())
        detachFile();
    THOR_THROW_IF_FALSE(this->fileName.empty());
    this->fileName = fileName;

    this->fileAccessRequirement = fileAccessRequirement;

    int32_t o_flags = O_CLOEXEC;
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
        THOR_THROW_IF_FALSE(fileDescriptor >= 0);
    }
    ownsFileDescriptor = true;

    // GpuDirect Storage takes pointers to its parameters:
    this->fileOffset = fileOffset;
}

// With existing file descriptor
void Tensor::attachFile(const std::string &fileName,
                        const off_t fileOffset,
                        const FileAccess fileAccessRequirement,
                        int32_t fileDescriptor) {
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU ||
                        getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    if (!this->fileName.empty())
        detachFile();
    THOR_THROW_IF_FALSE(this->fileName.empty());
    this->fileName = fileName;

    this->fileAccessRequirement = fileAccessRequirement;

    this->fileDescriptor = fileDescriptor;
    ownsFileDescriptor = false;

    this->fileOffset = fileOffset;
}

void Tensor::detachFile() {
    if (fileName.empty()) {
        return;
    }

    if (ownsFileDescriptor)
        close(fileDescriptor);

    fileName.clear();
}

void Tensor::copyObject(const Tensor &other) {
    *((ReferenceCounted *)this) = *((ReferenceCounted *)&other);

    placement = other.placement;
    mem = other.mem;
    storageElementOffset = other.storageElementOffset;
    storageNumElements = other.storageNumElements;
    customStridesElements = other.customStridesElements;
    instanceId = other.instanceId;

    descriptor = other.descriptor;
    descriptorOverridden = other.descriptorOverridden;
    overriddenDescriptor = other.overriddenDescriptor;
    cpuMemPinnedViaCudaHostRegister = other.cpuMemPinnedViaCudaHostRegister;
}

void Tensor::destroy() {
    detachFile();  // detachFile is a NOP when there is no attached file

    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        if (cpuMemPinnedViaCudaHostRegister) {
            cudaHostUnregister(mem);
            free(mem);
            cpuMemPinnedViaCudaHostRegister = false;
            mem = nullptr;
        } else {
            cudaError_t cudaStatus = cudaFreeHost(mem);
            mem = nullptr;
            if (cudaStatus != cudaSuccess) {
                printf("cuda status %d\n", cudaStatus);
                fflush(stdout);
            }
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        }

    } else if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        ScopedGpu scopedGpu(getPlacement().getDeviceNum());

        cudaError_t cudaStatus = cudaFree(mem);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
    } else {
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
                            placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
    mem = nullptr;
}

static uint64_t checkedWholeByteElementSizeBytes(TensorDescriptor::DataType dataType, const char *context) {
    const float size = TensorDescriptor::getElementSizeInBytes(dataType);
    if (size < 1.0f || size != std::floor(size)) {
        throw std::runtime_error(std::string(context) + " does not support packed sub-byte element types.");
    }
    return static_cast<uint64_t>(size);
}

static uint8_t *dataPointerWithElementOffset(void *mem, TensorDescriptor::DataType dataType, uint64_t elementOffset) {
    if (elementOffset == 0) {
        return static_cast<uint8_t *>(mem);
    }
    const uint64_t elementSizeBytes = checkedWholeByteElementSizeBytes(dataType, "Tensor storage offsets");
    return static_cast<uint8_t *>(mem) + elementOffset * elementSizeBytes;
}

static const uint8_t *dataPointerWithElementOffset(const void *mem, TensorDescriptor::DataType dataType, uint64_t elementOffset) {
    if (elementOffset == 0) {
        return static_cast<const uint8_t *>(mem);
    }
    const uint64_t elementSizeBytes = checkedWholeByteElementSizeBytes(dataType, "Tensor storage offsets");
    return static_cast<const uint8_t *>(mem) + elementOffset * elementSizeBytes;
}

static std::vector<uint64_t> denseStridesForDims(const std::vector<uint64_t>& dims) {
    if (dims.empty()) {
        throw std::runtime_error("Tensor dense strides require non-empty dimensions.");
    }
    std::vector<uint64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        if (strides[static_cast<size_t>(i) + 1] > std::numeric_limits<uint64_t>::max() / dims[static_cast<size_t>(i) + 1]) {
            throw std::runtime_error("Tensor dense stride computation overflowed.");
        }
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] * dims[static_cast<size_t>(i) + 1];
    }
    return strides;
}

static uint64_t maxElementOffsetForView(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
    if (dims.size() != strides.size() || dims.empty()) {
        throw std::runtime_error("Tensor alias view dimensions/strides must have the same non-zero rank.");
    }
    uint64_t maxOffset = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 0) {
            throw std::runtime_error("Tensor alias view dimensions must be non-zero.");
        }
        const uint64_t extentMinusOne = dims[i] - 1;
        if (extentMinusOne != 0 && strides[i] > std::numeric_limits<uint64_t>::max() / extentMinusOne) {
            throw std::runtime_error("Tensor alias view stride extent overflowed.");
        }
        const uint64_t axisMax = extentMinusOne * strides[i];
        if (maxOffset > std::numeric_limits<uint64_t>::max() - axisMax) {
            throw std::runtime_error("Tensor alias view max offset overflowed.");
        }
        maxOffset += axisMax;
    }
    return maxOffset;
}

template <typename ElementDataType>
ElementDataType *Tensor::getMemPtr() {
    THOR_THROW_IF_FALSE(isInitialized());
    THOR_THROW_IF_FALSE(mem != nullptr);

    using BaseT = std::remove_cv_t<ElementDataType>;
    static_assert(!std::is_const_v<ElementDataType>, "Non-const getMemPtr() should not return const pointer type");

    // Ensure that if the convenience template parameter ElementDataType is used that it agrees with the descriptor
    if (!(is_same<BaseT, void>::value)) {
        if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
            THOR_THROW_IF_FALSE((is_same<BaseT, half>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
            THOR_THROW_IF_FALSE((is_same<BaseT, float>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP64)
            THOR_THROW_IF_FALSE((is_same<BaseT, double>::value));
        else if (is_same<BaseT, char>::value)
            THOR_THROW_IF_FALSE(descriptor.getDataType() == TensorDescriptor::DataType::UINT8 ||
                                descriptor.getDataType() == TensorDescriptor::DataType::INT8);
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
            THOR_THROW_IF_FALSE((is_same<BaseT, int8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
            THOR_THROW_IF_FALSE((is_same<BaseT, int16_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
            THOR_THROW_IF_FALSE((is_same<BaseT, int32_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT64)
            THOR_THROW_IF_FALSE((is_same<BaseT, int64_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint16_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint32_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT64)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint64_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
            THOR_THROW_IF_FALSE((is_same<BaseT, bool>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::BF16)
            THOR_THROW_IF_FALSE((is_same<BaseT, __nv_bfloat16>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E4M3)
            THOR_THROW_IF_FALSE((is_same<BaseT, __nv_fp8_e4m3>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E5M2)
            THOR_THROW_IF_FALSE((is_same<BaseT, __nv_fp8_e5m2>::value));
        else
            THOR_UNREACHABLE();
    }
    return reinterpret_cast<ElementDataType *>(dataPointerWithElementOffset(mem, descriptor.getDataType(), storageElementOffset));
}

template <typename ElementDataType>
const ElementDataType *Tensor::getMemPtr() const {
    THOR_THROW_IF_FALSE(isInitialized());
    THOR_THROW_IF_FALSE(mem != nullptr);

    using BaseT = std::remove_cv_t<ElementDataType>;

    // Ensure that if the convenience template parameter ElementDataType is used that it agrees with the descriptor
    if (!(is_same<BaseT, void>::value)) {
        if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
            THOR_THROW_IF_FALSE((is_same<BaseT, half>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
            THOR_THROW_IF_FALSE((is_same<BaseT, float>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP64)
            THOR_THROW_IF_FALSE((is_same<BaseT, double>::value));
        else if (is_same<BaseT, char>::value)
            THOR_THROW_IF_FALSE(descriptor.getDataType() == TensorDescriptor::DataType::UINT8 ||
                                descriptor.getDataType() == TensorDescriptor::DataType::INT8);
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
            THOR_THROW_IF_FALSE((is_same<BaseT, int8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
            THOR_THROW_IF_FALSE((is_same<BaseT, int16_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
            THOR_THROW_IF_FALSE((is_same<BaseT, int32_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::INT64)
            THOR_THROW_IF_FALSE((is_same<BaseT, int64_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint16_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint32_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT64)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint64_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
            THOR_THROW_IF_FALSE((is_same<BaseT, bool>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
            THOR_THROW_IF_FALSE((is_same<BaseT, uint8_t>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::BF16)
            THOR_THROW_IF_FALSE((is_same<BaseT, __nv_bfloat16>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E4M3)
            THOR_THROW_IF_FALSE((is_same<BaseT, __nv_fp8_e4m3>::value));
        else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E5M2)
            THOR_THROW_IF_FALSE((is_same<BaseT, __nv_fp8_e5m2>::value));
        else
            THOR_UNREACHABLE();
    }

    return reinterpret_cast<const ElementDataType *>(
        dataPointerWithElementOffset(mem, descriptor.getDataType(), storageElementOffset));
}

template <typename ElementDataType>
ElementDataType Tensor::getElement(vector<unsigned long> dimensionIndex) {
    THOR_THROW_IF_FALSE(!uninitialized());

#ifdef THOR_DEBUG
    // This seems like too much overhead to get just one element, so it is explicitly removed for release.
    if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, half>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, float>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, double>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int64_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint64_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, bool>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BF16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_bfloat16>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E4M3)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_fp8_e4m3>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E5M2)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_fp8_e5m2>::value));
    else
        THOR_UNREACHABLE();
#endif

    THOR_THROW_IF_FALSE(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    return *getElementPointer<ElementDataType>(dimensionIndex);
}

template <typename ElementDataType>
void Tensor::setElement(std::vector<unsigned long> dimensionIndex, const ElementDataType &value) {
    THOR_THROW_IF_FALSE(!uninitialized());

#ifdef THOR_DEBUG
    // This seems like too much overhead to get just one element, so it is explicitly removed for release.
    if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, half>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, float>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, double>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int64_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint64_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, bool>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BF16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_bfloat16>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E4M3)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_fp8_e4m3>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E5M2)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_fp8_e5m2>::value));
    else
        THOR_UNREACHABLE();
#endif

    THOR_THROW_IF_FALSE(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    *getElementPointer<ElementDataType>(dimensionIndex) = value;
}

template <typename ElementDataType>
ElementDataType *Tensor::getElementPointer(std::vector<unsigned long> dimensionIndex) {
    THOR_THROW_IF_FALSE(!uninitialized());

#ifdef THOR_DEBUG
    // This seems like too much overhead to get just one element, so it is explicitly removed for release.
    if (descriptor.getDataType() == TensorDescriptor::DataType::FP16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, half>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, float>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, double>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT8)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::INT64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, int64_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT8)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint16_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT32)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint32_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::UINT64)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint64_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BOOLEAN)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, bool>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, uint8_t>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::BF16)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_bfloat16>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E4M3)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_fp8_e4m3>::value));
    else if (descriptor.getDataType() == TensorDescriptor::DataType::FP8_E5M2)
        THOR_THROW_IF_FALSE((is_same<ElementDataType, __nv_fp8_e5m2>::value));
    else
        THOR_UNREACHABLE();
#endif

    THOR_THROW_IF_FALSE(getDescriptor().getDataType() != TensorDescriptor::DataType::PACKED_BOOLEAN);
    const std::vector<uint64_t> dims = getDimensions();
    const std::vector<uint64_t> strides = getStridesElements();
    THOR_THROW_IF_FALSE(dimensionIndex.size() <= dims.size());
    uint64_t elementOffset = storageElementOffset;
    for (size_t i = 0; i < dimensionIndex.size(); ++i) {
        THOR_THROW_IF_FALSE(dimensionIndex[i] < dims[i]);
        elementOffset += dimensionIndex[i] * strides[i];
    }
    return reinterpret_cast<ElementDataType *>(dataPointerWithElementOffset(mem, descriptor.getDataType(), elementOffset));
}

// Use same memory, but change dimension sizes, must be exactly the same number of elements.
void Tensor::reshape(vector<unsigned long> dimensions) {
    descriptor.reshape(dimensions);
    if (customStridesElements.empty()) {
        // TensorDescriptor::reshape preserves the original descriptor stride cache. Keep view-stride
        // queries correct by computing dense strides from the current visible dimensions on demand.
        return;
    }
    customStridesElements = denseStridesForDims(dimensions);
}

Tensor Tensor::aliasView(vector<unsigned long> dimensions, vector<unsigned long> strides_elements, uint64_t element_offset) const {
    THOR_THROW_IF_FALSE(isInitialized());
    (void)checkedWholeByteElementSizeBytes(descriptor.getDataType(), "Tensor::aliasView");
    if (dimensions.empty() || dimensions.size() != strides_elements.size()) {
        throw std::runtime_error("Tensor::aliasView requires dimensions and strides with the same non-zero rank.");
    }
    for (uint64_t dim : dimensions) {
        if (dim == 0) {
            throw std::runtime_error("Tensor::aliasView dimensions must be non-zero.");
        }
    }
    const uint64_t maxRelativeOffset = maxElementOffsetForView(dimensions, strides_elements);
    if (element_offset > std::numeric_limits<uint64_t>::max() - maxRelativeOffset) {
        throw std::runtime_error("Tensor::aliasView offset overflowed.");
    }
    const uint64_t totalVisibleMaxOffset = element_offset + maxRelativeOffset;
    if (storageElementOffset > std::numeric_limits<uint64_t>::max() - totalVisibleMaxOffset) {
        throw std::runtime_error("Tensor::aliasView base offset overflowed.");
    }
    const uint64_t sourceAllocationElements = storageNumElements;
    if (storageElementOffset + totalVisibleMaxOffset >= sourceAllocationElements) {
        throw std::runtime_error("Tensor::aliasView would address beyond the source tensor allocation.");
    }

    Tensor view = *this;
    view.descriptor = TensorDescriptor(descriptor.getDataType(), dimensions);
    view.storageElementOffset = storageElementOffset + element_offset;
    view.customStridesElements = std::move(strides_elements);
    return view;
}

std::vector<uint64_t> Tensor::getStridesElements() const {
    THOR_THROW_IF_FALSE(isInitialized());
    if (!customStridesElements.empty()) {
        return customStridesElements;
    }
    return denseStridesForDims(getDimensions());
}

bool Tensor::isDenseContiguous() const {
    THOR_THROW_IF_FALSE(isInitialized());
    return getStridesElements() == denseStridesForDims(getDimensions());
}

// Change the dimensions of the tensor, possibly changing the amount of memory used.
// Frees the old memory and uses a new, uninitialized block of memory.
void Tensor::resize(vector<unsigned long> dimensions) {
    descriptor = TensorDescriptor(descriptor.getDataType(), dimensions);
    destroy();
    storageElementOffset = 0;
    storageNumElements = descriptor.getTotalNumElements();
    customStridesElements.clear();
    allocateMemory();
}

// Stream is on either the source or dest device
void Tensor::copyFromAsync(Tensor source, Stream stream) {
    THOR_THROW_IF_FALSE(!uninitialized());
    vector<int> devicesInvolved;
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(source.getPlacement().getDeviceNum());
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(getPlacement().getDeviceNum());
    if (!devicesInvolved.empty())
        THOR_THROW_IF_FALSE(stream.getGpuNum() == devicesInvolved[0] ||
                            (devicesInvolved.size() == 2 && stream.getGpuNum() == devicesInvolved[1]));
    copyFromAsync(source, stream, true);
}

void Tensor::downloadSection(Tensor &source, Stream &stream, uint64_t sourceOffset, uint64_t destOffset, uint64_t sizeBytes) {
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        THOR_THROW_IF_FALSE(stream.getGpuNum() == source.getPlacement().getDeviceNum());
    }

    // Check that access is within range
    uint64_t destArraySizeBytes = descriptor.getArraySizeInBytes();
    uint64_t sourceArraySizeBytes = source.descriptor.getArraySizeInBytes();
    THOR_THROW_IF_FALSE(destOffset + sizeBytes <= destArraySizeBytes);
    THOR_THROW_IF_FALSE(sourceOffset + sizeBytes <= sourceArraySizeBytes);

    uint8_t *destMemBytes = static_cast<uint8_t *>(getMemPtr<void>());
    uint8_t *sourceMemBytes = static_cast<uint8_t *>(source.getMemPtr<void>());
    cudaError_t cudaStatus =
        cudaMemcpyAsync(destMemBytes + destOffset, sourceMemBytes + sourceOffset, sizeBytes, cudaMemcpyDeviceToHost, stream.getStream());
    THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
}

void Tensor::uploadSection(Tensor &dest, Stream &stream, uint64_t sourceOffset, uint64_t destOffset, uint64_t sizeBytes) {
    if (dest.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        THOR_THROW_IF_FALSE(stream.getGpuNum() == dest.getPlacement().getDeviceNum());
    }

    // Check that access is within range
    uint64_t sourceArraySizeBytes = descriptor.getArraySizeInBytes();
    uint64_t destArraySizeBytes = dest.descriptor.getArraySizeInBytes();
    THOR_THROW_IF_FALSE(sourceOffset + sizeBytes <= sourceArraySizeBytes);
    THOR_THROW_IF_FALSE(destOffset + sizeBytes <= destArraySizeBytes);

    uint8_t *sourceMemBytes = static_cast<uint8_t *>(getMemPtr<void>());
    uint8_t *destMemBytes = static_cast<uint8_t *>(dest.getMemPtr<void>());
    cudaError_t cudaStatus =
        cudaMemcpyAsync(destMemBytes + destOffset, sourceMemBytes + sourceOffset, sizeBytes, cudaMemcpyHostToDevice, stream.getStream());
    THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
}

// If the tensor changes datatypes such that the size changes, then stream must be on the device with the larger tensor size.
// Otherwise stream may be on either device
void Tensor::moveFromAsync(Tensor source, Stream stream) {
    THOR_THROW_IF_FALSE(!uninitialized());
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU ||
                        getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!fileName.empty());
    THOR_THROW_IF_FALSE(fileAccessRequirement != FileAccess::READ_ONLY);

    vector<int> devicesInvolved;
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(source.getPlacement().getDeviceNum());
    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU)
        devicesInvolved.push_back(getPlacement().getDeviceNum());
    if (!devicesInvolved.empty())
        THOR_THROW_IF_FALSE(stream.getGpuNum() == devicesInvolved[0] ||
                            (devicesInvolved.size() == 2 && stream.getGpuNum() == devicesInvolved[1]));
    copyFromAsync(source, stream, false);
}

struct CheckIoBytesParams : HostFunctionArgsBase {
    CheckIoBytesParams(const ssize_t expectedBytes, const ssize_t *actualBytes) : expectedBytes(expectedBytes), actualBytes(actualBytes) {}
    const ssize_t expectedBytes;
    const ssize_t *actualBytes;
};

void checkBytesIoOp(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    THOR_THROW_IF_FALSE(baseParams != nullptr);
    CheckIoBytesParams *checkIoBytesParams = dynamic_cast<CheckIoBytesParams *>(baseParams);
    THOR_THROW_IF_FALSE(checkIoBytesParams != nullptr);

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

static void read_exact_at(int fd, uint8_t *p, size_t total, off_t offset, const std::string &fileName) {
    size_t left = total;

    while (left > 0) {
        ssize_t n = pread(fd, p, left, offset);

        if (n > 0) {
            p += static_cast<size_t>(n);
            left -= static_cast<size_t>(n);
            offset += static_cast<off_t>(n);
            continue;
        }

        if (n == 0) {
            throw std::runtime_error("EOF reading " + fileName + " (wanted " + std::to_string(total) + " bytes, got " +
                                     std::to_string(total - left) + ")");
        }

        int e = errno;
        if (e == EINTR)
            continue;
        if (e == EAGAIN)
            continue;

        throw std::runtime_error("pread failed for " + fileName + " at offset " + std::to_string(static_cast<long long>(offset)) +
                                 " (wanted " + std::to_string(left) + " bytes): " + std::string(strerror(e)));
    }
}

void performRead(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    THOR_THROW_IF_FALSE(baseParams != nullptr);
    PerformReadParams *performReadParams = dynamic_cast<PerformReadParams *>(baseParams);
    THOR_THROW_IF_FALSE(performReadParams != nullptr);

    uint8_t *memPtr = (uint8_t *)performReadParams->memPtr;
    const size_t totalBytesToRead = performReadParams->totalBytesToRead;
    const string fileName = performReadParams->fileName;
    const off_t fileOffset = performReadParams->fileOffset;
    const int32_t fileDescriptor = performReadParams->fileDescriptor;

    read_exact_at(fileDescriptor, memPtr, totalBytesToRead, fileOffset, fileName);
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
    THOR_THROW_IF_FALSE(baseParams != nullptr);
    PerformWriteParams *performWriteParams = dynamic_cast<PerformWriteParams *>(baseParams);
    THOR_THROW_IF_FALSE(performWriteParams != nullptr);

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

void Tensor::loadFromFile(Stream stream, std::optional<uint32_t> crc) {
    THOR_THROW_IF_FALSE(
        !crc.has_value());  // Need to solve for this. Not using GDS now so not solving it. ArchiveReader checks crc internally now.

    THOR_THROW_IF_FALSE(!uninitialized());
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU ||
                        getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!fileName.empty());
    THOR_THROW_IF_FALSE(fileAccessRequirement != FileAccess::WRITE_ONLY);

    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        // Can't deallocate the bounce buffer in stream.enqueueHostFunction(), so this process is synchronous.
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor bounceBuffer = clone(cpuPlacement);
        // disk -> cpu
        PerformReadParams args(bounceBuffer.getMemPtr(), getArraySizeInBytes(), fileName, fileOffset, fileDescriptor);
        performRead(&args);
        // cpu -> gpu
        copyFromAsync(bounceBuffer, stream);
        Event copyDoneEvent = stream.putEvent(false, true);
        copyDoneEvent.synchronize();
    } else {
        std::unique_ptr<HostFunctionArgsBase> args(
            new PerformReadParams(getMemPtr(), getArraySizeInBytes(), fileName, fileOffset, fileDescriptor));
        stream.enqueueHostFunction(performRead, std::move(args));
    }
}

void Tensor::dumpToFile(Stream stream) {
    THOR_THROW_IF_FALSE(!uninitialized());
    THOR_THROW_IF_FALSE(getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU ||
                        getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!fileName.empty());
    THOR_THROW_IF_FALSE(fileAccessRequirement != FileAccess::READ_ONLY);

    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        // Since no gpu direct, allocate a cpu buffer, copy data into that buffer, copy data from buffer to gpu.
        // I can't do this asynchronously because I need to free the host buffer once it is finished and can't call cuda functions
        // on a stream enqueued host function.
        // So when gpuDirectStorage is not available and enabled, this will be a synchronous operation.
        // It could be made async if I am ok keeping the other tensor alive.
        // Can't deallocate the bounce buffer in stream.enqueueHostFunction(), so this process is synchronous.
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor bounceBuffer = clone(cpuPlacement);
        // gpu -> cpu
        bounceBuffer.copyFromAsync(*this, stream);
        Event copyDoneEvent = stream.putEvent(false, true);
        copyDoneEvent.synchronize();
        // cpu -> disk
        PerformWriteParams args(bounceBuffer.getMemPtr(), getArraySizeInBytes(), fileName, fileOffset, fileDescriptor);
        performWrite(&args);
    } else {
        std::unique_ptr<HostFunctionArgsBase> args(
            new PerformWriteParams(getMemPtr(), getArraySizeInBytes(), fileName, fileOffset, fileDescriptor));
        stream.enqueueHostFunction(performWrite, std::move(args));
    }
}

TensorDescriptor Tensor::getDescriptor() const {
    THOR_THROW_IF_FALSE(!uninitialized());

    if (descriptorOverridden)
        return overriddenDescriptor;
    return descriptor;
}

void Tensor::overrideDescriptor(TensorDescriptor overrideDescriptor) {
    THOR_THROW_IF_FALSE(!uninitialized());

    descriptorOverridden = true;
    overriddenDescriptor = overrideDescriptor;
}

void Tensor::clearDescriptorOverride() {
    THOR_THROW_IF_FALSE(!uninitialized());
    descriptorOverridden = false;
}

void Tensor::copyFromAsync(Tensor source, Stream copyStream, bool mustPreserveSourceValue) {
    THOR_THROW_IF_FALSE(!uninitialized());
    THOR_THROW_IF_FALSE(!source.uninitialized());

    if (source.getTensorId() == getTensorId() && source.getDescriptor().getDataType() == getDescriptor().getDataType()) {
        return;
    }

    cudaError_t cudaStatus;
    THOR_THROW_IF_FALSE(copyStream.isInitialized());

    // must have the same number of elements
    TensorDescriptor sourceDescriptor = source.getDescriptor();
    TensorDescriptor destDescriptor = getDescriptor();
    if (sourceDescriptor.getTotalNumElements() != destDescriptor.getTotalNumElements()) {
        printf("Error: total number of elements does not match when copying tensors.\n source dimensions %s\n dest dimensions %s\n",
               source.dimensionsToString().c_str(),
               dimensionsToString().c_str());
        fflush(stdout);
    }
    THOR_THROW_IF_FALSE(sourceDescriptor.getTotalNumElements() == destDescriptor.getTotalNumElements());

    int sourceDeviceNum = source.placement.getDeviceNum();
    int destDeviceNum = placement.getDeviceNum();

    // Handle data type conversions that also need a placement change.
    //
    // CPU->CPU conversion intentionally stays in the ordinary CPU copy path below.
    //
    // If the destination data type is larger than the source data type, then this is supported by copying the
    // source-sized value to the destination placement and then up-converting in-place.
    //
    // If the destination data type is smaller than the source data type, preserving copyFromAsync intentionally
    // rejects the operation on the C++ side.  Supporting that case without mutating the source requires a hidden
    // temporary, which can mask an unexpected slow path in performance-sensitive C++ code.  The Python binding
    // implements that as an explicit binding-level convenience instead.  Non-preserving move-style copies may
    // still down-convert the source in-place before moving the narrower value.
    if (sourceDescriptor.getDataType() != destDescriptor.getDataType() && source.placement != placement) {
        if (sourceDescriptor.getArraySizeInBytes() <= destDescriptor.getArraySizeInBytes()) {
            if (placement.getMemDevice() == TensorPlacement::MemDevices::GPU)
                THOR_THROW_IF_FALSE(placement.getDeviceNum() == copyStream.getGpuNum());

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
            if (mustPreserveSourceValue) {
                throw runtime_error(
                    "Tensor::copyFromAsync refuses preserving cross-placement copies where the destination data type is "
                    "smaller than the source data type because that requires a hidden temporary; use an explicit "
                    "staging buffer (fast) or a temporary (slow) when this path is intended");
            }

            if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU)
                THOR_THROW_IF_FALSE(source.placement.getDeviceNum() == copyStream.getGpuNum());

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
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
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
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        ScopedGpu scopedGpu(sourceDeviceNum);

        cudaStatus =
            cudaMemcpyAsync(mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyDeviceToHost, copyStream.getStream());
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
    } else if (source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
               placement.getMemDevice() == TensorPlacement::MemDevices::GPU) {
        if (destDeviceNum == sourceDeviceNum) {
            // Local copy
            ScopedGpu scopedGpu(destDeviceNum);

            if (sourceDescriptor.getDataType() == destDescriptor.getDataType()) {
                cudaStatus = cudaMemcpyAsync(
                    mem, source.mem, sourceDescriptor.getArraySizeInBytes(), cudaMemcpyDeviceToDevice, copyStream.getStream());
                THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
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
            THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
        }
    } else {
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
                            placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(source.placement.getMemDevice() == TensorPlacement::MemDevices::CPU ||
                            source.placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    }
}

void Tensor::memset(int8_t value, uint64_t numElements) {
    // On GPU this would require device synchronization so this is not supported.
    THOR_THROW_IF_FALSE(placement.getMemDevice() != TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::CPU);

    uint64_t numBytes;
    if (numElements == 0) {
        numBytes = getArraySizeInBytes();
    } else {
        if (getDataType() == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            THOR_THROW_IF_FALSE(numElements % 8 == 0);
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

template <typename T>
inline T castCpuTensorValue(double value) {
    return T(value);
}

template <>
inline half castCpuTensorValue<half>(double value) {
    return half(static_cast<float>(value));
}

template <>
inline __nv_bfloat16 castCpuTensorValue<__nv_bfloat16>(double value) {
    return __float2bfloat16(static_cast<float>(value));
}

template <>
inline __nv_fp8_e4m3 castCpuTensorValue<__nv_fp8_e4m3>(double value) {
    return __nv_fp8_e4m3(static_cast<float>(value));
}

template <>
inline __nv_fp8_e5m2 castCpuTensorValue<__nv_fp8_e5m2>(double value) {
    return __nv_fp8_e5m2(static_cast<float>(value));
}

template <typename T>
static void fillCpuIdentityMatrixOnesTyped(Tensor tensor) {
    uint64_t N = tensor.getDimensions()[0];
    T *mem = tensor.getMemPtr<T>();
    const T one = castCpuTensorValue<T>(1.0);
    for (uint64_t i = 0; i < N; ++i)
        mem[i * N + i] = one;
}

void fillCpuIdentityMatrixOnes(void *data) {
    HostFunctionArgsBase *baseArgs = static_cast<HostFunctionArgsBase *>(data);
    THOR_THROW_IF_FALSE(baseArgs != nullptr);
    IdentityMatrixArgs *args = dynamic_cast<IdentityMatrixArgs *>(baseArgs);
    THOR_THROW_IF_FALSE(args != nullptr);

    THOR_THROW_IF_FALSE(args->tensor.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    TensorDescriptor::DataType dataType = args->tensor.getDataType();
    THOR_THROW_IF_FALSE(dataType != TensorDescriptor::DataType::PACKED_BOOLEAN);

    if (dataType == TensorDescriptor::DataType::FP16)
        fillCpuIdentityMatrixOnesTyped<half>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::BF16)
        fillCpuIdentityMatrixOnesTyped<__nv_bfloat16>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::FP8_E4M3)
        fillCpuIdentityMatrixOnesTyped<__nv_fp8_e4m3>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::FP8_E5M2)
        fillCpuIdentityMatrixOnesTyped<__nv_fp8_e5m2>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::FP32)
        fillCpuIdentityMatrixOnesTyped<float>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::FP64)
        fillCpuIdentityMatrixOnesTyped<double>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::INT8)
        fillCpuIdentityMatrixOnesTyped<int8_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::INT16)
        fillCpuIdentityMatrixOnesTyped<int16_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::INT32)
        fillCpuIdentityMatrixOnesTyped<int32_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::INT64)
        fillCpuIdentityMatrixOnesTyped<int64_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::UINT8)
        fillCpuIdentityMatrixOnesTyped<uint8_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::UINT16)
        fillCpuIdentityMatrixOnesTyped<uint16_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::UINT32)
        fillCpuIdentityMatrixOnesTyped<uint32_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::UINT64)
        fillCpuIdentityMatrixOnesTyped<uint64_t>(args->tensor);
    else if (dataType == TensorDescriptor::DataType::BOOLEAN)
        fillCpuIdentityMatrixOnesTyped<bool>(args->tensor);
    else
        THOR_UNREACHABLE();
}

Tensor Tensor::identityMatrix(uint32_t N, TensorPlacement placement, TensorDescriptor::DataType dataType, Stream stream) {
    THOR_THROW_IF_FALSE(dataType != TensorDescriptor::DataType::PACKED_BOOLEAN);
    Tensor tensor(placement, TensorDescriptor(dataType, {N, N}));

    if (placement.getMemDevice() == TensorPlacement::MemDevices::CPU) {
        tensor.memsetAsync(stream, 0);
        std::unique_ptr<HostFunctionArgsBase> args(new IdentityMatrixArgs(tensor));
        stream.enqueueHostFunction(fillCpuIdentityMatrixOnes, std::move(args));
    } else {
        tensor.memsetAsync(stream, 0);
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
    THOR_THROW_IF_FALSE(baseArgs != nullptr);
    MemsetArgs *args = dynamic_cast<MemsetArgs *>(baseArgs);
    THOR_THROW_IF_FALSE(args != nullptr);
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
                THOR_THROW_IF_FALSE(numElements % 8 == 0);
                numBytes = numElements / 8;
            } else {
                numBytes = numElements * (getArraySizeInBytes() / getTotalNumElements());
            }
        }

        ScopedGpu scopedGpu(placement.getDeviceNum());
        cudaError_t cudaStatus;
        cudaStatus = cudaMemsetAsync(mem, value, numBytes, stream);
        THOR_THROW_IF_FALSE(cudaStatus == cudaSuccess);
    } else {
        std::unique_ptr<HostFunctionArgsBase> args(new MemsetArgs(*this, value, numElements));
        stream.enqueueHostFunction(callMemsetOnTensor, std::move(args));
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

struct FillRandomArgs : HostFunctionArgsBase {
    FillRandomArgs(Tensor tensor, double minValue, double maxValue) : tensor(tensor), minValue(minValue), maxValue(maxValue) {}
    Tensor tensor;
    double minValue;
    double maxValue;
};

template <typename T>
static void fillCpuRandomFloating(Tensor tensor, double minValue, double maxValue, uint32_t numProcs, uint64_t elementsPerThread) {
    const uint64_t numElements = tensor.getTotalNumElements();
    T *mem = tensor.getMemPtr<T>();
    if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
        {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_real_distribution<double> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = castCpuTensorValue<T>(dis(gen));
            }
        }
    } else {
        random_device rd;
        uint32_t seed = Tensor::getThreadIdHash(rd());
        mt19937 gen(seed);
        uniform_real_distribution<double> dis(minValue, maxValue);

        for (uint64_t i = 0; i < numElements; ++i) {
            mem[i] = castCpuTensorValue<T>(dis(gen));
        }
    }
}

template <typename T, typename DISTRIBUTION_VALUE_TYPE>
static void fillCpuRandomIntegral(Tensor tensor,
                                  DISTRIBUTION_VALUE_TYPE minValue,
                                  DISTRIBUTION_VALUE_TYPE maxValue,
                                  uint32_t numProcs,
                                  uint64_t elementsPerThread) {
    const uint64_t numElements = tensor.getTotalNumElements();
    T *mem = tensor.getMemPtr<T>();
    if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
        {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<DISTRIBUTION_VALUE_TYPE> dis(minValue, maxValue);

#pragma omp for schedule(static, elementsPerThread)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = static_cast<T>(dis(gen));
            }
        }
    } else {
        random_device rd;
        uint32_t seed = Tensor::getThreadIdHash(rd());
        mt19937 gen(seed);
        uniform_int_distribution<DISTRIBUTION_VALUE_TYPE> dis(minValue, maxValue);

        for (uint64_t i = 0; i < numElements; ++i) {
            mem[i] = static_cast<T>(dis(gen));
        }
    }
}

static void fillCpuRandomPackedBoolean(Tensor tensor) {
    uint64_t numElements = (tensor.getTotalNumElements() + 7) / 8;
    const uint32_t numProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
    const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;

    uint8_t *mem = tensor.getMemPtr<uint8_t>();
    if (numProcs > 1) {
#pragma omp parallel num_threads(numProcs)
        {
            random_device rd;
            uint32_t seed = Tensor::getThreadIdHash(rd());
            mt19937 gen(seed);
            uniform_int_distribution<uint16_t> dis(0, 255);

#pragma omp for schedule(static, elementsPerThread)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = static_cast<uint8_t>(dis(gen));
            }
        }
    } else {
        random_device rd;
        uint32_t seed = Tensor::getThreadIdHash(rd());
        mt19937 gen(seed);
        uniform_int_distribution<uint16_t> dis(0, 255);

        for (uint64_t i = 0; i < numElements; ++i) {
            mem[i] = static_cast<uint8_t>(dis(gen));
        }
    }
}

void fillCpuRandom(void *data) {
    HostFunctionArgsBase *baseArgs = static_cast<HostFunctionArgsBase *>(data);
    THOR_THROW_IF_FALSE(baseArgs != nullptr);
    FillRandomArgs *args = dynamic_cast<FillRandomArgs *>(baseArgs);
    THOR_THROW_IF_FALSE(args != nullptr);

    THOR_THROW_IF_FALSE(args->tensor.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
    TensorDescriptor::DataType dataType = args->tensor.getDataType();

    Tensor tensor = args->tensor;
    double minValue = args->minValue;
    double maxValue = args->maxValue;
    if (minValue > maxValue)
        swap(minValue, maxValue);
    uint64_t numElements = tensor.getTotalNumElements();
    const uint32_t numProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
    const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;

    if (dataType == TensorDescriptor::DataType::FP16)
        fillCpuRandomFloating<half>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::BF16)
        fillCpuRandomFloating<__nv_bfloat16>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP8_E4M3)
        fillCpuRandomFloating<__nv_fp8_e4m3>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP8_E5M2)
        fillCpuRandomFloating<__nv_fp8_e5m2>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP32)
        fillCpuRandomFloating<float>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP64)
        fillCpuRandomFloating<double>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT8)
        fillCpuRandomIntegral<int8_t, int16_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT16)
        fillCpuRandomIntegral<int16_t, int32_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT32)
        fillCpuRandomIntegral<int32_t, int64_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT64)
        fillCpuRandomIntegral<int64_t, int64_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT8)
        fillCpuRandomIntegral<uint8_t, uint16_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT16)
        fillCpuRandomIntegral<uint16_t, uint32_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT32)
        fillCpuRandomIntegral<uint32_t, uint64_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT64)
        fillCpuRandomIntegral<uint64_t, uint64_t>(tensor, minValue, maxValue, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::BOOLEAN)
        fillCpuRandomIntegral<bool, uint16_t>(tensor, 0, 1, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN)
        fillCpuRandomPackedBoolean(tensor);
    else
        THOR_UNREACHABLE();
}

static void adjustSignedIntegralRandomRange(double& minValue, double& maxValue) {
    if (maxValue > 0) {
        // integer rounding (truncation) rounds away from maxValue in this case
        if (maxValue == int64_t(maxValue))
            maxValue += 0.99999;
    }
    if (minValue < 0) {
        // integer rounding (truncation) rounds away from minValue in this case
        if (minValue == int64_t(minValue))
            minValue -= 0.99999;
    }
}

static void adjustUnsignedIntegralRandomRange(double& maxValue) {
    if (maxValue == uint64_t(maxValue))
        maxValue += 0.99999;
}

void Tensor::fillRandom(double minValue, double maxValue, Stream stream) {
    if (maxValue < minValue)
        swap(maxValue, minValue);

    if (getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        std::unique_ptr<HostFunctionArgsBase> args(new FillRandomArgs(*this, minValue, maxValue));
        stream.enqueueHostFunction(fillCpuRandom, std::move(args));
    } else {
        TensorDescriptor::DataType dataType = getDataType();
        if (dataType == TensorDescriptor::DataType::FP16) {
            launchGpuFillRandom<half>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::BF16) {
            launchGpuFillRandom<__nv_bfloat16>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::FP8_E4M3) {
            launchGpuFillRandom<__nv_fp8_e4m3>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::FP8_E5M2) {
            launchGpuFillRandom<__nv_fp8_e5m2>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::FP32) {
            launchGpuFillRandom<float>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::FP64) {
            launchGpuFillRandom<double>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT8) {
            // Since for int I am rounding down for integer types, adding just under 1.0 to the continuous distribution gives
            // the range [minValue, maxValue] where minValue and maxValue have equal likelihood of being generated to all other values.
            // Note that converting to an integer truncates and so rounds toward 0.
            adjustSignedIntegralRandomRange(minValue, maxValue);
            launchGpuFillRandom<int8_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT16) {
            adjustSignedIntegralRandomRange(minValue, maxValue);
            launchGpuFillRandom<int16_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT32) {
            adjustSignedIntegralRandomRange(minValue, maxValue);
            launchGpuFillRandom<int32_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::INT64) {
            adjustSignedIntegralRandomRange(minValue, maxValue);
            launchGpuFillRandom<int64_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT8) {
            adjustUnsignedIntegralRandomRange(maxValue);
            launchGpuFillRandom<uint8_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT16) {
            adjustUnsignedIntegralRandomRange(maxValue);
            launchGpuFillRandom<uint16_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT32) {
            adjustUnsignedIntegralRandomRange(maxValue);
            launchGpuFillRandom<uint32_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::UINT64) {
            adjustUnsignedIntegralRandomRange(maxValue);
            launchGpuFillRandom<uint64_t>(getMemPtr(), getTotalNumElements(), minValue, maxValue, stream);
        } else if (dataType == TensorDescriptor::DataType::BOOLEAN) {
            launchGpuFillRandom<bool>(getMemPtr(), getTotalNumElements(), 0, 1.999999, stream);
        } else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            launchGpuFillRandom<uint8_t>(getMemPtr(), (getTotalNumElements() + 7) / 8, 0, 255.999999, stream);
        } else {
            THOR_UNREACHABLE();
        }
    }
}

void Tensor::fillZero(Stream dataStream) { this->fill(0.0, dataStream); }

struct CpuFillParams : HostFunctionArgsBase {
    CpuFillParams(double value, Tensor tensor) : value(value), tensor(tensor) {}

    double value;
    Tensor tensor;
};

template <typename T>
static void fillValueTyped(Tensor tensor, double rawValue, uint32_t numProcs, uint64_t elementsPerThread) {
    uint64_t numElements = tensor.getTotalNumElements();
    T *mem = tensor.getMemPtr<T>();
    T value = castCpuTensorValue<T>(rawValue);
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

void fillValue(void *params) {
    HostFunctionArgsBase *baseParams = static_cast<HostFunctionArgsBase *>(params);
    THOR_THROW_IF_FALSE(baseParams != nullptr);
    CpuFillParams *cpuFillParams = dynamic_cast<CpuFillParams *>(baseParams);
    THOR_THROW_IF_FALSE(cpuFillParams != nullptr);

    uint64_t numElements = cpuFillParams->tensor.getTotalNumElements();
    const uint32_t numProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
    const uint64_t elementsPerThread = (numElements + (numProcs - 1)) / numProcs;

    TensorDescriptor::DataType dataType = cpuFillParams->tensor.getDataType();

    if (dataType == TensorDescriptor::DataType::FP16)
        fillValueTyped<half>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::BF16)
        fillValueTyped<__nv_bfloat16>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP8_E4M3)
        fillValueTyped<__nv_fp8_e4m3>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP8_E5M2)
        fillValueTyped<__nv_fp8_e5m2>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP32)
        fillValueTyped<float>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::FP64)
        fillValueTyped<double>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT8)
        fillValueTyped<int8_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT16)
        fillValueTyped<int16_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT32)
        fillValueTyped<int32_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::INT64)
        fillValueTyped<int64_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT8)
        fillValueTyped<uint8_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT16)
        fillValueTyped<uint16_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT32)
        fillValueTyped<uint32_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::UINT64)
        fillValueTyped<uint64_t>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::BOOLEAN)
        fillValueTyped<bool>(cpuFillParams->tensor, cpuFillParams->value, numProcs, elementsPerThread);
    else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
        uint8_t *mem = cpuFillParams->tensor.getMemPtr<uint8_t>();
        uint8_t value = cpuFillParams->value ? 0b11111111 : 0;
        numElements = (numElements + 7) / 8;
        const uint32_t packedNumProcs = max(min((uint64_t)omp_get_num_procs(), numElements / 500000), uint64_t(1));
        const uint64_t packedElementsPerThread = (numElements + (packedNumProcs - 1)) / packedNumProcs;
        if (packedNumProcs > 1) {
#pragma omp parallel for schedule(static, packedElementsPerThread) shared(mem, value, packedElementsPerThread, numElements) default(none)
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        } else {
            for (uint64_t i = 0; i < numElements; ++i) {
                mem[i] = value;
            }
        }
    } else {
        THOR_UNREACHABLE();
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
                castCpuTensorValue<half>(value), (half *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::BF16) {
            launchFillValueGpuKernel<__nv_bfloat16>(castCpuTensorValue<__nv_bfloat16>(value),
                                                   (__nv_bfloat16 *)getMemPtr(),
                                                   getTotalNumElements(),
                                                   getPlacement().getDeviceNum(),
                                                   stream);
        } else if (dataType == TensorDescriptor::DataType::FP8_E4M3) {
            launchFillValueGpuKernel<__nv_fp8_e4m3>(castCpuTensorValue<__nv_fp8_e4m3>(value),
                                                   (__nv_fp8_e4m3 *)getMemPtr(),
                                                   getTotalNumElements(),
                                                   getPlacement().getDeviceNum(),
                                                   stream);
        } else if (dataType == TensorDescriptor::DataType::FP8_E5M2) {
            launchFillValueGpuKernel<__nv_fp8_e5m2>(castCpuTensorValue<__nv_fp8_e5m2>(value),
                                                   (__nv_fp8_e5m2 *)getMemPtr(),
                                                   getTotalNumElements(),
                                                   getPlacement().getDeviceNum(),
                                                   stream);
        } else if (dataType == TensorDescriptor::DataType::FP32) {
            launchFillValueGpuKernel<float>(value, (float *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::FP64) {
            launchFillValueGpuKernel<double>(value, (double *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT8) {
            launchFillValueGpuKernel<uint8_t>(value, (uint8_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT16) {
            launchFillValueGpuKernel<uint16_t>(
                value, (uint16_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT32) {
            launchFillValueGpuKernel<uint32_t>(
                value, (uint32_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::UINT64) {
            launchFillValueGpuKernel<uint64_t>(
                value, (uint64_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT8) {
            launchFillValueGpuKernel<int8_t>(value, (int8_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT16) {
            launchFillValueGpuKernel<int16_t>(value, (int16_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT32) {
            launchFillValueGpuKernel<int32_t>(value, (int32_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::INT64) {
            launchFillValueGpuKernel<int64_t>(value, (int64_t *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::BOOLEAN) {
            launchFillValueGpuKernel<bool>(value, (bool *)getMemPtr(), getTotalNumElements(), getPlacement().getDeviceNum(), stream);
        } else if (dataType == TensorDescriptor::DataType::PACKED_BOOLEAN) {
            uint8_t packedValue = 0;
            if (value)
                packedValue = 0b11111111;
            uint64_t numPackedBytes = (getTotalNumElements() + 7) / 8;
            launchFillValueGpuKernel<uint8_t>(packedValue, (uint8_t *)getMemPtr(), numPackedBytes, getPlacement().getDeviceNum(), stream);
        } else {
            THOR_UNREACHABLE();
        }
    }
}

Tensor Tensor::transposeMatrix(Stream stream) {
    vector<uint64_t> dimensions = getDimensions();
    // Generally the transpose of a higher order tensor would be any permutation of the tensor's dimensions, in that case the particular
    // permutation would also need to be specified. I'm not doing that now and unless some need arises I probably won't implement that.
    THOR_THROW_IF_FALSE(dimensions.size() == 2);
    vector<uint64_t> transposedDimensions;
    transposedDimensions.push_back(dimensions[1]);
    transposedDimensions.push_back(dimensions[0]);
    Tensor transposedTensor = clone(transposedDimensions);

    if (getDataType() == TensorDescriptor::DataType::FP16) {
        matrixTranspose((half *)transposedTensor.getMemPtr(), (half *)getMemPtr(), dimensions[0], dimensions[1], stream);
    } else if (getDataType() == TensorDescriptor::DataType::FP32) {
        matrixTranspose((float *)transposedTensor.getMemPtr(), (float *)getMemPtr(), dimensions[0], dimensions[1], stream);
    } else {
        THOR_UNREACHABLE();  // TODO
    }

    return transposedTensor;
}

// void Tensor::transposeSquareMatrixInPlace(Stream stream) {
//     vector<uint64_t> dimensions = getDimensions();
//     // Generally the transpose of a higher order tensor would be any permutation of the tensor's dimensions, in that case the particular
//     // permutation would also need to be specified. I'm not doing that now and unless some need arises I probably won't implement that.
//     THOR_THROW_IF_FALSE(dimensions.size() == 2);
//     THOR_THROW_IF_FALSE(dimensions[0] == dimensions[1]);
//
//     if (getDataType() == TensorDescriptor::DataType::FP16) {
//         matrixTransposeSquare((half *)getMemPtr(), (half *)getMemPtr(), dimensions[0], stream);
//     } else if (getDataType() == TensorDescriptor::DataType::FP32) {
//         matrixTransposeSquare((float *)getMemPtr(), (float *)getMemPtr(), dimensions[0], stream);
//     } else {
//         THOR_UNREACHABLE();  // TODO
//     }
// }

template void *Tensor::getMemPtr();
template half *Tensor::getMemPtr();
template float *Tensor::getMemPtr();
template double *Tensor::getMemPtr();
template int8_t *Tensor::getMemPtr();
template int16_t *Tensor::getMemPtr();
template int32_t *Tensor::getMemPtr();
template int64_t *Tensor::getMemPtr();
template uint8_t *Tensor::getMemPtr();
template uint16_t *Tensor::getMemPtr();
template uint32_t *Tensor::getMemPtr();
template uint64_t *Tensor::getMemPtr();
template char *Tensor::getMemPtr();
template __nv_bfloat16 *Tensor::getMemPtr();
template __nv_fp8_e4m3 *Tensor::getMemPtr();
template __nv_fp8_e5m2 *Tensor::getMemPtr();

template const void *Tensor::getMemPtr<void>() const;
template const half *Tensor::getMemPtr<half>() const;
template const float *Tensor::getMemPtr<float>() const;
template const double *Tensor::getMemPtr<double>() const;
template const int8_t *Tensor::getMemPtr<int8_t>() const;
template const int16_t *Tensor::getMemPtr<int16_t>() const;
template const int32_t *Tensor::getMemPtr<int32_t>() const;
template const int64_t *Tensor::getMemPtr<int64_t>() const;
template const uint8_t *Tensor::getMemPtr<uint8_t>() const;
template const uint16_t *Tensor::getMemPtr<uint16_t>() const;
template const uint32_t *Tensor::getMemPtr<uint32_t>() const;
template const uint64_t *Tensor::getMemPtr<uint64_t>() const;
template const char *Tensor::getMemPtr<char>() const;
template const __nv_bfloat16 *Tensor::getMemPtr<__nv_bfloat16>() const;
template const __nv_fp8_e4m3 *Tensor::getMemPtr<__nv_fp8_e4m3>() const;
template const __nv_fp8_e5m2 *Tensor::getMemPtr<__nv_fp8_e5m2>() const;

template half Tensor::getElement(vector<unsigned long> dimensionIndex);
template float Tensor::getElement(vector<unsigned long> dimensionIndex);
template double Tensor::getElement(vector<unsigned long> dimensionIndex);
template int8_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template int16_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template int32_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template int64_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint8_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint16_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint32_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template uint64_t Tensor::getElement(vector<unsigned long> dimensionIndex);
template char Tensor::getElement(vector<unsigned long> dimensionIndex);
template __nv_bfloat16 Tensor::getElement(vector<unsigned long> dimensionIndex);
template __nv_fp8_e4m3 Tensor::getElement(vector<unsigned long> dimensionIndex);
template __nv_fp8_e5m2 Tensor::getElement(vector<unsigned long> dimensionIndex);

template void Tensor::setElement(vector<unsigned long> dimensionIndex, const half &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const float &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const double &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int8_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int16_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int32_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const int64_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint8_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint16_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint32_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const uint64_t &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const char &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const __nv_bfloat16 &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const __nv_fp8_e4m3 &value);
template void Tensor::setElement(vector<unsigned long> dimensionIndex, const __nv_fp8_e5m2 &value);

template void *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template half *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template float *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template double *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int8_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int16_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int32_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template int64_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint8_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint16_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint32_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template uint64_t *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template char *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template __nv_bfloat16 *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template __nv_fp8_e4m3 *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
template __nv_fp8_e5m2 *Tensor::getElementPointer(vector<unsigned long> dimensionIndex);
