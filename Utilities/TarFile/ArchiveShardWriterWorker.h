#pragma once

#include "Crc32.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/TarArchive.h"
#include "Utilities/TarFile/TarHeaderHelper.h"
#include "Utilities/TarFile/UringDirect.h"

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

/*
 Async read from gpu memory (cuda to pinned cpu buffer) and async write to disk (io_uring in O_DIRECT mode).
 Double buffered to prefetch.
 CRC32 zlib-ng zng_crc32_z parallel carryless multiply - it's free not a limiting factor.
 5 workers in a thread pool.
 Can hit upwards of 100 GBPS dumped to disk with memory bandwidth and storage hardware that can handle that.

 Other than pipelining async operations, the additional difficulty is that writes
 to disk must be in chunks of exactly 4KB (2^12) bytes (no option to pad) and files
 in tar must be multiples of exactly 512 bytes (or padded to make it so).

 The state machine below shows how this is handled. The leftover from the previous
 file that is not within a chunk of 4KB is forwarded as the tail and prepended to
 the next write operation. Tar ends with at least 2 512 byte blocks of 0's, but
 to satisfy the 4KB write chunk requirement, additional blocks will be added
 as necessary.

 The index is appended to the valid tar file, and is ignored by tar.

 State Machine:
  ____________________________________________________________________________________
 | Start    | Finish   | Start    | Finish   |      Section            | Containing  |
 | Loading  | Loading  | Dumping  | Dumping  |                         | Buffer      |
 -------------------------------------------------------------------------------------
 |        initial      | consume0 | consume1 | header                  |      0      |
 | initial  | consume0 | consume0 | consume1 | payload                 |      0      |
 |       consume0      | consume1 | consume0 | prior tail              |      1      |
 |       consume0      | consume1 | consume0 | prior pad + next header |      1      |
 | consume0 | consume1 | consume1 | consume0 | payload                 |      1      |
 |       consume1      | consume0 | consume1 | prior tail              |      0      |
 |       consume1      | consume0 | consume1 | prior pad + next header |      0      |
 | consume1 | consume0 | consume0 | consume1 | payload                 |      0      |
 |       consume0      | consume1 | consume0 | prior tail              |      1      |
 |       consume0      | consume1 | consume0 | prior pad + next header |      1      |
 | consume0 | consume1 | consume1 | consume0 | payload                 |      1      |
 |       consume1      | consume0 | consume1 | prior tail              |      0      |
 |       consume1      | consume0 | consume1 | prior pad + next header |      0      |
 | consume1 | consume0 | consume0 | consume1 | payload                 |      0      |
 |       consume0      | consume1 | consume0 | prior tail              |      1      |
 |       consume0      | consume1 | consume0 | prior pad + next header |      1      |
 | consume0 | consume1 | consume1 | consume0 | payload                 |      1      |
 |       consume1      | consume0 | consume0 | prior tail              |      0      |
 |       consume1      | consume0 | consume0 | prior pad + next header |      0      |
 | consume1 | post     | post     | post     | payload                 |      0      |
 |       post          | post     | post     | prior tail              |      0      |
 |       post          | post     | post     | prior pad + tar ending  |      0      |
 -------------------------------------------------------------------------------------

 */

struct ArchiveFileWriteParams {
    ThorImplementation::Tensor deviceTensor;

    uint64_t offsetBytes;
    uint64_t numBytes;
    std::string path_in_tar;
};

enum class WriterState {
    INITIAL,
    CONSUME_BUFFER_0,
    CONSUME_BUFFER_1,
    POST,
};

class ArchiveShardWriterWorker {
   public:
    ArchiveShardWriterWorker() : uringDirect(64) {
        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {fiveHundredMBPlusTail});
        bounceBuffer[0] = ThorImplementation::Tensor(cpuPlacement, descriptor, 4096);
        bounceBuffer[1] = ThorImplementation::Tensor(cpuPlacement, descriptor, 4096);
        uringDirect.registerReusableBuffers({bounceBuffer[0].getMemPtr(), bounceBuffer[1].getMemPtr()},
                                            {fiveHundredMBPlusTail, fiveHundredMBPlusTail});
        bounceBufferMem[0] = bounceBuffer[0].getMemPtr<uint8_t>();
        bounceBufferMem[1] = bounceBuffer[1].getMemPtr<uint8_t>();
    }

    void process(std::vector<ArchiveFileWriteParams>& plan, std::string archiveShardPath, std::vector<uint32_t>& crcs) {
        assert(plan.size() > 0);
        uringDirect.registerDumpFile(archiveShardPath);
        uint32_t numCompletionsToFinish[2] = {0, 0};
        constexpr uint64_t fourKLeftoverMask = 0xFFF;

        crcs.clear();
        crcs.reserve(plan.size());

        uint32_t finalFetchingBuffer;
        dumpedFileOffsetBytes = 0;
        WriterState state = WriterState::INITIAL;

        for (uint32_t i = 0; i < plan.size(); ++i) {
            if (state == WriterState::INITIAL) {
                // Ensure I have the needed stream
                uint32_t deviceNum = plan[0].deviceTensor.getPlacement().getDeviceNum();
                // Get existing or put one there if missing:
                auto [it, inserted] = streams.try_emplace(deviceNum, Stream::getNextDownloadStream(deviceNum));
                Stream& stream = it->second;

                // set numTailBytes to the tar header length for the first file:
                numTailPadAndHeaderBytes = createTarSeparator(plan[0].path_in_tar, plan[0].numBytes, 0, 0, tailAndTarSeparator, thirtyTwoK);
                // Copy the header to the front of the bounce buffer
                std::memcpy(bounceBufferMem[0], tailAndTarSeparator, numTailPadAndHeaderBytes);

                // Initiate fetch of the first chunk of the file, place it on the bounce buffer after the tar header:
                assert(plan[0].numBytes <= fiveHundredMB);
                bounceBufferReady[0] = prefetchGpuBuffer(
                    bounceBuffer[0], plan[0].deviceTensor, stream, plan[0].offsetBytes, plan[0].numBytes, numTailPadAndHeaderBytes);

                if (plan.size() == 1) {
                    finalFetchingBuffer = 0;
                    state = WriterState::POST;
                } else {
                    state = WriterState::CONSUME_BUFFER_0;
                }
            } else {
                assert(state == WriterState::CONSUME_BUFFER_0 || state == WriterState::CONSUME_BUFFER_1);

                uint32_t loadedBuffer;
                uint32_t dumpingBuffer;
                if (state == WriterState::CONSUME_BUFFER_0) {
                    loadedBuffer = 0;
                    dumpingBuffer = 1;
                } else {
                    loadedBuffer = 1;
                    dumpingBuffer = 0;
                }

                // Ensure I have the needed stream
                uint32_t deviceNum = plan[i].deviceTensor.getPlacement().getDeviceNum();
                // Get existing or put one there if missing:
                auto [it, inserted] = streams.try_emplace(deviceNum, Stream::getNextDownloadStream(deviceNum));
                Stream& stream = it->second;

                // Wait for done prefetch loadedBuffer from GPU
                bounceBufferReady[loadedBuffer].synchronize();

                // Compute the number of tail bytes that will be left over,
                // after dumping everything in loadedBuffer to disk (the bytes that are not in a chunk of size 4KB bytes).
                uint32_t loadedBufferTailPadAndHeaderBytes = numTailPadAndHeaderBytes;
                uint32_t totalLoadedBufferBytes = plan[i - 1].numBytes + loadedBufferTailPadAndHeaderBytes;
                uint32_t fourKAlignedBytes = numFourKAlignedBytes(totalLoadedBufferBytes);
                uint32_t numTailBytesToForward = totalLoadedBufferBytes - fourKAlignedBytes;

                // Initiate dump of loadedBuffer to disk
                numCompletionsToFinish[loadedBuffer] = dumpBufferToArchiveFile(loadedBuffer, fourKAlignedBytes);

                // Forward the unaligned bytes from the back of loadedBuffer into the tailAndTarSeparator scratch memory
                if (numTailBytesToForward > 0)
                    std::memcpy(tailAndTarSeparator, bounceBufferMem[loadedBuffer] + fourKAlignedBytes, numTailBytesToForward);

                // Consider the size of the previous file, pad to multiple of 512 bytes as needed.
                // Create the header for the next file.
                // Append both of these onto the scratch buffer (tailAndTarSeparator),
                // after the tail bytes that are already at the front of it.
                numTailPadAndHeaderBytes = createTarSeparator(
                    plan[i].path_in_tar, plan[i].numBytes, plan[i - 1].numBytes, numTailBytesToForward, tailAndTarSeparator, thirtyTwoK);

                // Wait for dump of buffer 1 to disk to complete  - it is possible that there has not yet been a dump from that buffer
                if (numCompletionsToFinish[dumpingBuffer] > 0)
                    uringDirect.waitCompletionsInOrder(numCompletionsToFinish[dumpingBuffer]);
                numCompletionsToFinish[dumpingBuffer] = 0;

                // Copy the scratch buffer to the head of the next bounce buffer to load the next chunk into.
                std::memcpy(bounceBufferMem[dumpingBuffer], tailAndTarSeparator, numTailPadAndHeaderBytes);

                // Initiate prefetch buffer 1 from GPU, starting immediately past the prepended tail, pad and header.
                bounceBufferReady[dumpingBuffer] = prefetchGpuBuffer(bounceBuffer[dumpingBuffer],
                                                                     plan[i].deviceTensor,
                                                                     stream,
                                                                     plan[i].offsetBytes,
                                                                     plan[i].numBytes,
                                                                     numTailPadAndHeaderBytes);

                // Compute CRC of the payload part of loadedBuffer
                uint32_t crc = crc32_ieee(
                    0xFFFFFFFF, (uint8_t*)bounceBufferMem[loadedBuffer] + loadedBufferTailPadAndHeaderBytes, plan[i - 1].numBytes);
                crcs.push_back(crc);

                if (plan.size() == i + 1) {
                    finalFetchingBuffer = dumpingBuffer;  // because pre-fetching into it now.
                    state = WriterState::POST;
                } else {
                    if (state == WriterState::CONSUME_BUFFER_0)
                        state = WriterState::CONSUME_BUFFER_1;
                    else
                        state = WriterState::CONSUME_BUFFER_0;
                }
            }
        }

        assert(state == WriterState::POST);
        uint32_t freeBuffer = finalFetchingBuffer == 0 ? 1 : 0;

        // Wait for done prefetch of final buffer from GPU
        bounceBufferReady[finalFetchingBuffer].synchronize();

        // Compute the number of tail bytes that will be left over,
        // after dumping everything in loadedBuffer to disk (the bytes that are not in a chunk of size 4KB bytes).
        uint32_t loadedBufferTailPadAndHeaderBytes = numTailPadAndHeaderBytes;
        uint32_t totalLoadedBufferBytes = plan.back().numBytes + loadedBufferTailPadAndHeaderBytes;
        uint32_t fourKAlignedBytes = numFourKAlignedBytes(totalLoadedBufferBytes);
        uint32_t numTailBytesToForward = totalLoadedBufferBytes - fourKAlignedBytes;

        // Initiate dump final buffer to disk
        numCompletionsToFinish[finalFetchingBuffer] = dumpBufferToArchiveFile(finalFetchingBuffer, fourKAlignedBytes);

        // Forward the unaligned bytes from the back of loadedBuffer into the tailAndTarSeparator scratch memory
        if (numTailBytesToForward > 0)
            std::memcpy(tailAndTarSeparator, bounceBufferMem[finalFetchingBuffer] + fourKAlignedBytes, numTailBytesToForward);

        // Consider the size of the previous file, pad to multiple of 512 bytes as needed.
        // Create the end of tar marker.
        // Append both of these onto the scratch buffer (tailAndTarSeparator),
        // after the tail bytes that are already at the front of it.
        // Ensuring that the total number of bytes to write is a multiple of 4K, this can be accomplished by increasing the size
        // of the end of tar marker.
        numTailPadAndHeaderBytes =
            appendTarEndOfArchive(plan[plan.size() - 1].numBytes, numTailBytesToForward, tailAndTarSeparator, thirtyTwoK);
        assert((numTailPadAndHeaderBytes & fourKLeftoverMask) == 0);

        // Compute CRC of the payload part of final buffer
        uint32_t crc = crc32_ieee(
            0xFFFFFFFF, (uint8_t*)bounceBufferMem[finalFetchingBuffer] + loadedBufferTailPadAndHeaderBytes, plan[plan.size() - 1].numBytes);
        crcs.push_back(crc);

        // Wait for dump from prior stage out of free buffer to finish
        if (numCompletionsToFinish[freeBuffer] > 0)
            uringDirect.waitCompletionsInOrder(numCompletionsToFinish[freeBuffer]);
        numCompletionsToFinish[freeBuffer] = 0;

        // Copy the scratch buffer to the head of the next bounce buffer - containing the final tail bytes and the tar end marker
        std::memcpy(bounceBufferMem[freeBuffer], tailAndTarSeparator, numTailPadAndHeaderBytes);
        numCompletionsToFinish[freeBuffer] = dumpBufferToArchiveFile(freeBuffer, numTailPadAndHeaderBytes);

        // Wait for dump of both buffers to disk to complete
        uringDirect.waitCompletionsInOrder(numCompletionsToFinish[finalFetchingBuffer]);
        numCompletionsToFinish[finalFetchingBuffer] = 0;
        uringDirect.waitCompletionsInOrder(numCompletionsToFinish[freeBuffer]);
        numCompletionsToFinish[freeBuffer] = 0;

        UringDirect::Completion c = uringDirect.finishDumpedFile(5789);
        if (c.responseCode != 0)
            throw std::runtime_error("io_uring returned responseCode = " + std::to_string(c.responseCode) + "when writing file " +
                                     archiveShardPath);
    }

    static Event prefetchGpuBuffer(ThorImplementation::Tensor& cpuBuffer,
                                   ThorImplementation::Tensor& deviceTensor,
                                   Stream& stream,
                                   uint64_t offsetBytes,
                                   uint64_t numBytes,
                                   uint32_t numTailBytes) {
        cpuBuffer.downloadSection(deviceTensor, stream, offsetBytes, numTailBytes, numBytes);
        return stream.putEvent(false, true);
    }

    uint32_t dumpBufferToArchiveFile(uint32_t bufferIndex, uint32_t num4kSegmentedBytes) {
        constexpr uint64_t fourKLeftoverMask = 0xFFF;
        constexpr uint32_t kChunkBytes = (uint32_t(1) << 24);  // 16 MiB

        assert(num4kSegmentedBytes > 0);
        assert((num4kSegmentedBytes & fourKLeftoverMask) == 0);
        assert(bufferIndex < 2);

        uint64_t fileBase = dumpedFileOffsetBytes;

        uint32_t submitted = 0;
        uint32_t numOps = 0;

        while (submitted < num4kSegmentedBytes) {
            uint32_t len = num4kSegmentedBytes - submitted;
            if (len > kChunkBytes)
                len = kChunkBytes;

            assert((len & fourKLeftoverMask) == 0);
            assert(((fileBase + submitted) & fourKLeftoverMask) == 0);
            assert((submitted & fourKLeftoverMask) == 0);

            // IMPORTANT: pass bufOffsetBytes=submitted so each chunk writes the correct slice of the bounce buffer
            uringDirect.submitWriteFixed(bufferIndex,
                                         fileBase + submitted,
                                         len,
                                         /*bufOffsetBytes=*/submitted);

            ++numOps;
            submitted += len;
        }

        uringDirect.submit();
        dumpedFileOffsetBytes += num4kSegmentedBytes;
        return numOps;
    }

    static uint32_t numFourKAlignedBytes(uint32_t totalBytes) { return totalBytes & fourKBAligned; }

   private:
    ThorImplementation::Tensor bounceBuffer[2];
    uint8_t* bounceBufferMem[2];
    Event bounceBufferReady[2];
    std::unordered_map<uint32_t, Stream> streams;

    uint32_t numTailPadAndHeaderBytes;
    uint8_t tailAndTarSeparator[thirtyTwoK];

    UringDirect uringDirect;
    uint64_t dumpedFileOffsetBytes;
};
