#pragma once

#include "Crc32.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TarFile/TarArchive.h"
#include "Utilities/TarFile/UringDirect.h"

struct ArchiveFileReadParams {
    ThorImplementation::Tensor deviceTensor;

    // Where to place the bytes in the destination device tensor:
    uint64_t deviceOffsetBytes;

    // Number of payload bytes to read for this shard:
    uint64_t numBytes;

    // Absolute byte offset in the archive file where the payload begins (NOT the tar header).
    uint64_t archivePayloadOffsetBytes;
};

enum class ReaderState {
    INITIAL,
    CONSUME_BUFFER_0,
    CONSUME_BUFFER_1,
    POST,
};

class ArchiveShardReaderWorker {
   public:
    ArchiveShardReaderWorker() : uringDirect(64) {
        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {fiveHundredMBPlusTail});

        bounceBuffer[0] = ThorImplementation::Tensor(cpuPlacement, descriptor, 4096);
        bounceBuffer[1] = ThorImplementation::Tensor(cpuPlacement, descriptor, 4096);

        uringDirect.registerReusableBuffers({bounceBuffer[0].getMemPtr(), bounceBuffer[1].getMemPtr()},
                                            {fiveHundredMBPlusTail, fiveHundredMBPlusTail});

        bounceBufferMem[0] = bounceBuffer[0].getMemPtr<uint8_t>();
        bounceBufferMem[1] = bounceBuffer[1].getMemPtr<uint8_t>();
    }

    // Reads each shard from archiveShardPath and uploads into the corresponding device tensor offsets.
    // Produces CRCs for the payload bytes in the same order as plan.
    void process(std::vector<ArchiveFileReadParams>& plan, const std::string& archiveShardPath, std::vector<uint32_t>& crcs) {
        assert(plan.size() > 0);

        // Symmetric to writer's registerDumpFile()
        uringDirect.registerLoadFile(archiveShardPath);

        crcs.clear();
        crcs.reserve(plan.size());

        // io_uring completion counts for each buffer
        uint32_t numCompletionsToFinish[2] = {0, 0};

        // GPU transfer events (so we don't overwrite a bounce buffer still being uploaded)
        Optional<Event> gpuTransferDone[2];

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
                assert(state == ReaderState::CONSUME_BUFFER_0 || state == ReaderState::CONSUME_BUFFER_1);

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
                crcs.push_back(crc);

                if (gpuTransferDone[loadingBuffer].isPresent())
                    gpuTransferDone[loadingBuffer].get().synchronize();

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

        assert(state == ReaderState::POST);
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
        crcs.push_back(crc);

        // Drain all outstanding GPU transfers
        uint32_t nonLastLoadingBuffer = lastLoadingBuffer == 0 ? 1 : 0;
        if (gpuTransferDone[nonLastLoadingBuffer].isPresent())
            gpuTransferDone[nonLastLoadingBuffer].get().synchronize();
        gpuTransferDone[lastLoadingBuffer].get().synchronize();
    }

    // Reads [fileOffset, fileOffset+numBytes) from archive into bounce buffer (O_DIRECT constraints handled by caller).
    // Returns number of io_uring operations submitted.
    uint32_t loadBufferFromArchiveFile(uint32_t bufferIndex, uint64_t fileOffsetBytes, uint32_t num4kBytes) {
        constexpr uint32_t kChunkBytes = (uint32_t(1) << 24);  // 16 MiB

        assert(bufferIndex < 2);
        assert(num4kBytes > 0);
        assert((num4kBytes & fourKBMask) == 0);
        assert((fileOffsetBytes & fourKBMask) == 0);

        uint32_t submitted = 0;
        uint32_t numOps = 0;

        while (submitted < num4kBytes) {
            uint32_t len = num4kBytes - submitted;
            if (len > kChunkBytes)
                len = kChunkBytes;

            assert((len & fourKBMask) == 0);
            assert(((fileOffsetBytes + submitted) & fourKBMask) == 0);
            assert((submitted & fourKBMask) == 0);

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
    // Schedules the disk read for one plan entry into bufferIndex, recording the per-buffer geometry
    // so the consumer knows what slice to upload and where.
    void scheduleReadIntoBuffer(uint32_t bufferIndex,
                                const ArchiveFileReadParams& p,
                                uint64_t prefixBytesForBuffer[2],
                                uint64_t payloadBytesForBuffer[2],
                                uint64_t dstDeviceOffsetForBuffer[2],
                                ThorImplementation::Tensor* dstDeviceTensorForBuffer[2],
                                uint32_t deviceNumForBuffer[2],
                                uint32_t numCompletionsToFinish[2]) {
        assert(bufferIndex < 2);
        assert(p.numBytes <= fiveHundredMB);

        const uint64_t payloadOffset = p.archivePayloadOffsetBytes;
        const uint64_t alignedFileOffset = alignDown4k(payloadOffset);
        const uint64_t prefixBytes = payloadOffset - alignedFileOffset;
        const uint64_t totalBytes = alignUp4k(prefixBytes + p.numBytes);

        if (totalBytes > fiveHundredMBPlusTail) {
            throw std::runtime_error("ArchiveShardReaderWorker: required read exceeds bounce buffer capacity");
        }

        prefixBytesForBuffer[bufferIndex] = prefixBytes;
        payloadBytesForBuffer[bufferIndex] = p.numBytes;
        dstDeviceOffsetForBuffer[bufferIndex] = p.deviceOffsetBytes;
        dstDeviceTensorForBuffer[bufferIndex] = const_cast<ThorImplementation::Tensor*>(&p.deviceTensor);

        const uint32_t dev = p.deviceTensor.getPlacement().getDeviceNum();
        deviceNumForBuffer[bufferIndex] = dev;

        // Submit the O_DIRECT aligned read into the bounce buffer.
        numCompletionsToFinish[bufferIndex] = loadBufferFromArchiveFile(bufferIndex, alignedFileOffset, (uint32_t)totalBytes);
    }

    Stream& getStreamForDevice(uint32_t deviceNum) {
        // Get existing or put one there if missing:
        auto [it, inserted] = streams.try_emplace(deviceNum, Stream::getNextDownloadStream(deviceNum));
        return it->second;
    }

    // Reverse of your downloadSection helper. Adjust the call if your API differs.
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

    UringDirect uringDirect;
};
