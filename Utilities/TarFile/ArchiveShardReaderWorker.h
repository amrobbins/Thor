#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

#include "Crc32.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TarFile/TarArchive.h"
#include "Utilities/TarFile/UringDirect.h"

namespace thor_file {

enum class ReaderState {
    INITIAL,
    CONSUME_BUFFER_0,
    CONSUME_BUFFER_1,
    POST,
};

class ArchiveShardReaderWorker {
   public:
    explicit ArchiveShardReaderWorker(ArchiveReaderContext context)
        : archiveDirectory(context.archiveDirectory), mtx(context.mtx), errorMessage(context.errorMessage), uringDirect(64, UringDirect::ioBackendFromEnv("THOR_TAR_READ_IO_BACKEND")) {
        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::TensorDescriptor descriptor(ThorImplementation::DataType::UINT8, {fiveHundredMBPlusTail});

        bounceBuffer[0] = ThorImplementation::Tensor(cpuPlacement, descriptor, 4096);
        bounceBuffer[1] = ThorImplementation::Tensor(cpuPlacement, descriptor, 4096);

        uringDirect.registerReusableBuffers({bounceBuffer[0].getMemPtr(), bounceBuffer[1].getMemPtr()},
                                            {fiveHundredMBPlusTail, fiveHundredMBPlusTail});

        bounceBufferMem[0] = bounceBuffer[0].getMemPtr<uint8_t>();
        bounceBufferMem[1] = bounceBuffer[1].getMemPtr<uint8_t>();
    }

    // Reads each shard from archiveShardPath and uploads into the corresponding device tensor offsets.
    // Produces CRCs for the payload bytes and compares against the crc in the plan.
    void process(ArchiveShardPlan job) {
        std::vector<ArchivePlanEntry>& plan = job.entries;
        THOR_THROW_IF_FALSE(plan.size() > 0);
        const std::string& archiveShardPath = (archiveDirectory / job.archiveShardPath).string();

        // Symmetric to writer's registerDumpFile()
        uringDirect.registerLoadFile(archiveShardPath);

        // io_uring completion counts for each buffer
        uint32_t numCompletionsToFinish[2] = {0, 0};

        // GPU transfer events (so we don't overwrite a bounce buffer still being uploaded)
        std::optional<Event> gpuTransferDone[2];

        // Keep the per-entry read geometry so we know what to upload when a buffer is ready
        uint64_t prefixBytesForBuffer[2] = {0, 0};
        uint64_t payloadBytesForBuffer[2] = {0, 0};
        uint64_t dstDeviceOffsetForBuffer[2] = {0, 0};
        ThorImplementation::Tensor* dstDeviceTensorForBuffer[2] = {nullptr, nullptr};
        uint32_t deviceNumForBuffer[2] = {0, 0};

        uint32_t lastLoadingBuffer;
        ReaderState state = ReaderState::INITIAL;

        for (uint32_t i = 0; i < plan.size(); ++i) {
            if (state == ReaderState::INITIAL) {
                // Schedule read of first entry into buffer 0
                scheduleReadIntoBuffer(0,
                                       plan[i],
                                       prefixBytesForBuffer,
                                       payloadBytesForBuffer,
                                       dstDeviceOffsetForBuffer,
                                       dstDeviceTensorForBuffer,
                                       deviceNumForBuffer,
                                       numCompletionsToFinish);

                if (plan.size() == 1) {
                    lastLoadingBuffer = 0;
                    state = ReaderState::POST;
                } else {
                    // Also schedule read of the second entry into buffer 1 *after* we begin consuming buffer 0
                    state = ReaderState::CONSUME_BUFFER_0;
                }
            } else {
                THOR_THROW_IF_FALSE(state == ReaderState::CONSUME_BUFFER_0 || state == ReaderState::CONSUME_BUFFER_1);

                uint32_t loadedBuffer;
                uint32_t loadingBuffer;
                if (state == ReaderState::CONSUME_BUFFER_0) {
                    loadedBuffer = 0;
                    loadingBuffer = 1;
                } else {
                    loadedBuffer = 1;
                    loadingBuffer = 0;
                }

                // Wait for disk read into loadedBuffer to complete
                if (numCompletionsToFinish[loadedBuffer] > 0) {
                    uringDirect.waitCompletionsInOrder(numCompletionsToFinish[loadedBuffer]);
                    numCompletionsToFinish[loadedBuffer] = 0;
                }

                // Upload payload slice from loadedBuffer to GPU (async)
                Stream& stream = getStreamForDevice(deviceNumForBuffer[loadedBuffer]);

                gpuTransferDone[loadedBuffer] = uploadGpuBuffer(bounceBuffer[loadedBuffer],
                                                                *dstDeviceTensorForBuffer[loadedBuffer],
                                                                stream,
                                                                prefixBytesForBuffer[loadedBuffer],
                                                                dstDeviceOffsetForBuffer[loadedBuffer],
                                                                payloadBytesForBuffer[loadedBuffer]);

                // Compute CRC of payload bytes (CPU side)
                const uint8_t* payloadPtr = bounceBufferMem[loadedBuffer] + prefixBytesForBuffer[loadedBuffer];
                uint32_t crc = crc32_ieee(0xFFFFFFFF, payloadPtr, (uint32_t)payloadBytesForBuffer[loadedBuffer]);
                if (crc != plan[i - 1].expectedCrc) {
                    synchronizeGpuTransfers(gpuTransferDone);
                    recordCrcMismatch(archiveShardPath, plan[i - 1], crc);
                    return;
                }

                if (gpuTransferDone[loadingBuffer].has_value())
                    gpuTransferDone[loadingBuffer].value().synchronize();

                scheduleReadIntoBuffer(loadingBuffer,
                                       plan[i],
                                       prefixBytesForBuffer,
                                       payloadBytesForBuffer,
                                       dstDeviceOffsetForBuffer,
                                       dstDeviceTensorForBuffer,
                                       deviceNumForBuffer,
                                       numCompletionsToFinish);

                if (i == plan.size() - 1) {
                    lastLoadingBuffer = loadingBuffer;
                    state = ReaderState::POST;
                } else {
                    state = (state == ReaderState::CONSUME_BUFFER_0) ? ReaderState::CONSUME_BUFFER_1 : ReaderState::CONSUME_BUFFER_0;
                }
            }
        }

        THOR_THROW_IF_FALSE(state == ReaderState::POST);
        if (numCompletionsToFinish[lastLoadingBuffer] > 0) {
            uringDirect.waitCompletionsInOrder(numCompletionsToFinish[lastLoadingBuffer]);
            numCompletionsToFinish[lastLoadingBuffer] = 0;
        }

        Stream& stream = getStreamForDevice(deviceNumForBuffer[lastLoadingBuffer]);
        gpuTransferDone[lastLoadingBuffer] = uploadGpuBuffer(bounceBuffer[lastLoadingBuffer],
                                                             *dstDeviceTensorForBuffer[lastLoadingBuffer],
                                                             stream,
                                                             prefixBytesForBuffer[lastLoadingBuffer],
                                                             dstDeviceOffsetForBuffer[lastLoadingBuffer],
                                                             payloadBytesForBuffer[lastLoadingBuffer]);

        const uint8_t* payloadPtr = bounceBufferMem[lastLoadingBuffer] + prefixBytesForBuffer[lastLoadingBuffer];
        uint32_t crc = crc32_ieee(0xFFFFFFFF, payloadPtr, (uint32_t)payloadBytesForBuffer[lastLoadingBuffer]);
        if (crc != plan.back().expectedCrc) {
            synchronizeGpuTransfers(gpuTransferDone);
            recordCrcMismatch(archiveShardPath, plan.back(), crc);
            return;
        }

        // Drain all outstanding GPU transfers
        uint32_t nonLastLoadingBuffer = lastLoadingBuffer == 0 ? 1 : 0;
        if (gpuTransferDone[nonLastLoadingBuffer].has_value())
            gpuTransferDone[nonLastLoadingBuffer].value().synchronize();
        gpuTransferDone[lastLoadingBuffer].value().synchronize();
    }

    // Reads [fileOffset, fileOffset+numBytes) from archive into bounce buffer (O_DIRECT constraints handled by caller).
    // Returns number of io_uring operations submitted.
    uint32_t loadBufferFromArchiveFile(uint32_t bufferIndex, uint64_t fileOffsetBytes, uint32_t num4kBytes) {
        constexpr uint32_t kChunkBytes = (uint32_t(1) << 24);  // 16 MiB

        THOR_THROW_IF_FALSE(bufferIndex < 2);
        THOR_THROW_IF_FALSE(num4kBytes > 0);
        THOR_THROW_IF_FALSE((num4kBytes & fourKBMask) == 0);
        THOR_THROW_IF_FALSE((fileOffsetBytes & fourKBMask) == 0);

        uint32_t submitted = 0;
        uint32_t numOps = 0;

        while (submitted < num4kBytes) {
            uint32_t len = num4kBytes - submitted;
            if (len > kChunkBytes)
                len = kChunkBytes;

            THOR_THROW_IF_FALSE((len & fourKBMask) == 0);
            THOR_THROW_IF_FALSE(((fileOffsetBytes + submitted) & fourKBMask) == 0);
            THOR_THROW_IF_FALSE((submitted & fourKBMask) == 0);

            uringDirect.submitReadFixed(bufferIndex, fileOffsetBytes + submitted, len, submitted);

            ++numOps;
            submitted += len;
        }

        uringDirect.submit();
        return numOps;
    }

    static uint64_t alignDown4k(uint64_t x) { return x & ~uint64_t(fourKBMask); }
    static uint64_t alignUp4k(uint64_t x) { return (x + fourKBMask) & ~uint64_t(fourKBMask); }

   private:
    static void synchronizeGpuTransfers(std::optional<Event> gpuTransferDone[2]) {
        for (uint32_t bufferIndex = 0; bufferIndex < 2; ++bufferIndex) {
            if (gpuTransferDone[bufferIndex].has_value()) {
                gpuTransferDone[bufferIndex].value().synchronize();
            }
        }
    }

    struct BufferedCrcProbe {
        std::optional<uint32_t> crc;
        std::string error;
    };

    static BufferedCrcProbe computeBufferedPreadCrc(const std::string& archiveShardPath,
                                                    uint64_t fileOffsetBytes,
                                                    uint64_t numBytes) {
        BufferedCrcProbe result;
        int fd = ::open(archiveShardPath.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            result.error = std::string("open failed: ") + std::strerror(errno);
            return result;
        }

        constexpr size_t kProbeChunkBytes = size_t{1} << 20;
        std::vector<uint8_t> buffer(kProbeChunkBytes);
        uint64_t bytesDone = 0;
        uint32_t crc = 0xFFFFFFFF;
        while (bytesDone < numBytes) {
            const size_t chunkBytes = static_cast<size_t>(
                std::min<uint64_t>(numBytes - bytesDone, static_cast<uint64_t>(buffer.size())));
            const uint64_t absoluteOffset = fileOffsetBytes + bytesDone;
            if (absoluteOffset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
                result.error = "file offset exceeds off_t range";
                ::close(fd);
                return result;
            }

            ssize_t response;
            do {
                response = ::pread(fd,
                                   buffer.data(),
                                   chunkBytes,
                                   static_cast<off_t>(absoluteOffset));
            } while (response < 0 && errno == EINTR);

            if (response < 0) {
                result.error = std::string("pread failed: ") + std::strerror(errno);
                ::close(fd);
                return result;
            }
            if (response == 0) {
                result.error = "pread reached EOF after " + std::to_string(bytesDone) +
                               " of " + std::to_string(numBytes) + " bytes";
                ::close(fd);
                return result;
            }

            crc = crc32_ieee(crc, buffer.data(), static_cast<size_t>(response));
            bytesDone += static_cast<uint64_t>(response);
        }

        ::close(fd);
        result.crc = crc;
        return result;
    }

    void recordCrcMismatch(const std::string& archiveShardPath,
                           const ArchivePlanEntry& entry,
                           uint32_t ioBackendCrc) {
        const BufferedCrcProbe buffered =
            computeBufferedPreadCrc(archiveShardPath, entry.fileOffsetBytes, entry.numBytes);

        std::ostringstream message;
        message << "CRC mismatch in file " << entry.pathInTar
                << " expected " << entry.expectedCrc
                << " actual " << ioBackendCrc
                << " io_backend=" << uringDirect.activeBackendName()
                << " archive_shard='" << archiveShardPath << "'"
                << " file_offset=" << entry.fileOffsetBytes
                << " size=" << entry.numBytes;
        if (buffered.crc.has_value()) {
            message << " buffered_pread_crc=" << buffered.crc.value();
            if (buffered.crc.value() == entry.expectedCrc) {
                message << " diagnosis=io_backend_read_corruption";
            } else if (buffered.crc.value() == ioBackendCrc) {
                message << " diagnosis=archive_payload_or_index_corruption";
            } else {
                message << " diagnosis=three_way_crc_disagreement";
            }
        } else {
            message << " buffered_pread_error='" << buffered.error << "'"
                    << " diagnosis=buffered_probe_failed";
        }

        std::lock_guard<std::mutex> lg(mtx);
        if (errorMessage.empty()) {
            errorMessage = message.str();
        }
    }

    // Schedules the disk read for one plan entry into bufferIndex, recording the per-buffer geometry
    // so the consumer knows what slice to upload and where.
    void scheduleReadIntoBuffer(uint32_t bufferIndex,
                                const ArchivePlanEntry& p,
                                uint64_t prefixBytesForBuffer[2],
                                uint64_t payloadBytesForBuffer[2],
                                uint64_t dstDeviceOffsetForBuffer[2],
                                ThorImplementation::Tensor* dstDeviceTensorForBuffer[2],
                                uint32_t deviceNumForBuffer[2],
                                uint32_t numCompletionsToFinish[2]) {
        THOR_THROW_IF_FALSE(bufferIndex < 2);
        THOR_THROW_IF_FALSE(p.numBytes <= fiveHundredMB);

        const uint64_t payloadOffset = p.fileOffsetBytes;
        const uint64_t alignedFileOffset = alignDown4k(payloadOffset);
        const uint64_t prefixBytes = payloadOffset - alignedFileOffset;
        const uint64_t totalBytes = alignUp4k(prefixBytes + p.numBytes);

        if (totalBytes > fiveHundredMBPlusTail) {
            throw std::runtime_error("ArchiveShardReaderWorker: required read exceeds bounce buffer capacity");
        }

        prefixBytesForBuffer[bufferIndex] = prefixBytes;
        payloadBytesForBuffer[bufferIndex] = p.numBytes;
        dstDeviceOffsetForBuffer[bufferIndex] = p.tensorOffsetBytes;
        dstDeviceTensorForBuffer[bufferIndex] = const_cast<ThorImplementation::Tensor*>(&p.tensor);

        const uint32_t dev = p.tensor.getPlacement().getDeviceNum();
        deviceNumForBuffer[bufferIndex] = dev;

        // Submit the O_DIRECT aligned read into the bounce buffer.
        numCompletionsToFinish[bufferIndex] = loadBufferFromArchiveFile(bufferIndex, alignedFileOffset, (uint32_t)totalBytes);
    }

    Stream& getStreamForDevice(uint32_t deviceNum) {
        // Get existing or put one there if missing:
        auto [it, inserted] = streams.try_emplace(deviceNum, Stream::getNextDownloadStream(deviceNum));
        return it->second;
    }

    static Event uploadGpuBuffer(ThorImplementation::Tensor& cpuBuffer,
                                 ThorImplementation::Tensor& deviceTensor,
                                 Stream& stream,
                                 uint64_t srcOffsetBytes,
                                 uint64_t dstOffsetBytes,
                                 uint64_t numBytes) {
        cpuBuffer.uploadSection(deviceTensor, stream, /*cpuOffsetBytes=*/srcOffsetBytes, /*deviceOffsetBytes=*/dstOffsetBytes, numBytes);
        return stream.putEvent(false, true);
    }

   private:
    ThorImplementation::Tensor bounceBuffer[2];
    uint8_t* bounceBufferMem[2];

    std::unordered_map<uint32_t, Stream> streams;

    const std::filesystem::path archiveDirectory;
    std::mutex& mtx;
    std::string& errorMessage;
    UringDirect uringDirect;
};

}  // namespace thor_file
