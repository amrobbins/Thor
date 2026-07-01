#pragma once

#include <liburing.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

class UringDirect {
   public:
    static constexpr std::size_t kAlign = 4096;

    enum class IoBackend {
        Auto,
        UringDirect,
        PreadDirect,
        PreadBuffered,
    };

    explicit UringDirect(unsigned queueDepth = 64, IoBackend backend = ioBackendFromEnv()) : requestedBackend_(backend) {
        fallbackQueueDepth_ = std::max(1u, queueDepth);
        initializeBackend(queueDepth);
    }

    ~UringDirect() {
        shutdownFallbackWorkersNoexcept();

        // Unregister in reverse order (best-effort).
        if (usesIoUring() && fileRegistered_) {
            (void)io_uring_unregister_files(&ring_);
            fileRegistered_ = false;
        }
        if (usesIoUring() && buffersRegistered_) {
            (void)io_uring_unregister_buffers(&ring_);
            buffersRegistered_ = false;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        if (ringInited_) {
            io_uring_queue_exit(&ring_);
            ringInited_ = false;
        }
    }

    UringDirect(const UringDirect&) = delete;
    UringDirect& operator=(const UringDirect&) = delete;

    UringDirect(UringDirect&& other) noexcept { moveFrom(std::move(other)); }
    UringDirect& operator=(UringDirect&& other) noexcept {
        if (this != &other) {
            this->~UringDirect();
            moveFrom(std::move(other));
        }
        return *this;
    }

    static IoBackend ioBackendFromEnv() {
        const char* raw = std::getenv("THOR_IO_BACKEND");
        if (raw == nullptr || *raw == '\0') {
            return IoBackend::Auto;
        }

        std::string value(raw);
        value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0; }), value.end());
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (value == "auto")
            return IoBackend::Auto;
        if (value == "uring_direct" || value == "io_uring" || value == "io_uring_direct")
            return IoBackend::UringDirect;
        if (value == "pread_direct" || value == "direct_pread" || value == "pread_odirect" || value == "pread_o_direct")
            return IoBackend::PreadDirect;
        if (value == "pread_buffered" || value == "buffered_pread" || value == "pread" || value == "buffered")
            return IoBackend::PreadBuffered;

        throw std::runtime_error("Unsupported THOR_IO_BACKEND='" + std::string(raw) +
                                 "'. Supported values: auto, uring_direct, pread_direct, pread_buffered.");
    }

    const char* requestedBackendName() const { return backendName(requestedBackend_); }
    const char* activeBackendName() const { return backendName(activeBackend_); }
    IoBackend requestedBackend() const { return requestedBackend_; }
    IoBackend activeBackend() const { return activeBackend_; }

#ifdef THOR_GTEST
    static void testSetIoUringQueueInitResult(std::optional<int> responseCode) { testIoUringQueueInitResult() = responseCode; }
    static void testSetDirectOpenUnavailable(bool unavailable) { testDirectOpenUnavailable() = unavailable; }
    static void testSetNextIoUringSubmissionByteLimit(std::optional<std::uint32_t> limitBytes) {
        std::lock_guard<std::mutex> guard(testIoUringSubmissionByteLimitMutex());
        testNextIoUringSubmissionByteLimit() = limitBytes;
    }
    static void testResetCompatibilityWarning() { compatibilityWarningEmitted() = false; }
    static void testResetFallbackWorkerBlock() {
        std::lock_guard<std::mutex> guard(testFallbackWorkerBlockMutex());
        testFallbackWorkerBlockEnabled() = false;
        testFallbackWorkerStartedCountValue() = 0;
        testFallbackWorkerBlockCv().notify_all();
    }
    static void testSetFallbackWorkerBlockEnabled(bool enabled) {
        std::lock_guard<std::mutex> guard(testFallbackWorkerBlockMutex());
        testFallbackWorkerBlockEnabled() = enabled;
        if (!enabled) {
            testFallbackWorkerBlockCv().notify_all();
        }
    }
    static bool testWaitForFallbackWorkerStartedCount(std::size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(testFallbackWorkerBlockMutex());
        return testFallbackWorkerBlockCv().wait_for(
            lock, timeout, [count] { return testFallbackWorkerStartedCountValue() >= count; });
    }
#endif

    struct BufferDesc {
        void* ptr = nullptr;
        std::size_t lenBytes = 0;  // must be multiple of 4k
    };

    // Register multiple reusable buffers (fixed buffers).
    //
    // Rules enforced here (to avoid -EINVAL later):
    // - ptr must be 4k aligned
    // - len must be >0 and a multiple of 4k
    //
    // This replaces any existing buffer registration.
    void registerReusableBuffers(const std::vector<BufferDesc>& bufs) {
        if (bufs.empty()) {
            throw std::runtime_error("registerReusableBuffers: bufs is empty");
        }

        for (std::size_t i = 0; i < bufs.size(); ++i) {
            if (!bufs[i].ptr) {
                throw std::runtime_error("registerReusableBuffers: null ptr at index " + std::to_string(i));
            }
            if (!isAligned(bufs[i].ptr, kAlign)) {
                throw std::runtime_error("registerReusableBuffers: ptr not 4k aligned at index " + std::to_string(i));
            }
            if (bufs[i].lenBytes == 0) {
                throw std::runtime_error("registerReusableBuffers: lenBytes is 0 at index " + std::to_string(i));
            }
            if ((bufs[i].lenBytes % kAlign) != 0) {
                throw std::runtime_error("registerReusableBuffers: lenBytes not multiple of 4k at index " + std::to_string(i));
            }
        }

        // If already registered, replace registration (must unregister first).
        if (buffersRegistered_) {
            if (usesIoUring()) {
                int rc = io_uring_unregister_buffers(&ring_);
                if (rc < 0) {
                    throw std::runtime_error(std::string("io_uring_unregister_buffers failed: ") + std::strerror(-rc));
                }
            }
            buffersRegistered_ = false;
        }

        iovecs_.clear();
        iovecs_.resize(bufs.size());
        for (std::size_t i = 0; i < bufs.size(); ++i) {
            iovecs_[i].iov_base = bufs[i].ptr;
            iovecs_[i].iov_len = bufs[i].lenBytes;
        }

        if (iovecs_.size() > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
            throw std::runtime_error("registerReusableBuffers: too many buffers");
        }

        if (!usesIoUring()) {
            buffersRegistered_ = true;
            return;
        }

        int rc = io_uring_register_buffers(&ring_, iovecs_.data(), static_cast<unsigned>(iovecs_.size()));
        if (rc < 0) {
            int e = -rc;
            if (isAutoMode() && isBackendAvailabilityErrno(e)) {
                fallbackFromIoUring("io_uring_register_buffers failed: " + std::string(std::strerror(e)), IoBackend::PreadDirect);
                buffersRegistered_ = true;
                return;
            }
            throw std::runtime_error(std::string("io_uring_register_buffers failed: ") + std::strerror(e));
        }
        buffersRegistered_ = true;
    }

    // Convenience: add/replace by passing spans/pairs.
    void registerReusableBuffers(const std::vector<void*>& ptrs, const std::vector<std::size_t>& lens) {
        if (ptrs.size() != lens.size()) {
            throw std::runtime_error("registerReusableBuffers: ptrs/lens size mismatch");
        }
        std::vector<BufferDesc> b;
        b.reserve(ptrs.size());
        for (std::size_t i = 0; i < ptrs.size(); ++i) {
            b.push_back(BufferDesc{ptrs[i], lens[i]});
        }
        registerReusableBuffers(b);
    }

    std::size_t numRegisteredBuffers() const { return buffersRegistered_ ? iovecs_.size() : 0; }

    // 2) Register a file descriptor for dumping a bunch of data.
    //
    // This does two things:
    // - opens the file with O_DIRECT (and other sane flags),
    // - registers it with io_uring so later we can use "fixed file" submissions.
    //
    // Requirements/notes:
    // - The true O_DIRECT requirements (alignment constraints) are enforced at I/O time
    //   by the kernel/filesystem. Here we ensure we opened with O_DIRECT successfully.
    // - We don't attempt to query device logical block size here yet; we standardize on 4k.
    void registerDumpFile(const std::string& path, bool truncate = true, mode_t mode = 0644) {
        closeRegisteredFile();

        int flags = O_WRONLY | O_CREAT | O_CLOEXEC;
        if (activeBackend_ != IoBackend::PreadBuffered)
            flags |= O_DIRECT;
        if (truncate)
            flags |= O_TRUNC;

        fd_ = openForBackend(path, flags, mode, /*hasMode=*/true);
        if (fd_ < 0 && isAutoMode() && activeBackend_ == IoBackend::UringDirect && (flags & O_DIRECT) &&
            isBackendAvailabilityErrno(errno)) {
            int e = errno;
            fallbackFromIoUring("open(O_DIRECT) failed for '" + path + "': " + std::string(std::strerror(e)), IoBackend::PreadBuffered);
            flags &= ~O_DIRECT;
            fd_ = openForBackend(path, flags, mode, /*hasMode=*/true);
        }
        if (fd_ < 0 && isAutoMode() && activeBackend_ == IoBackend::PreadDirect && (flags & O_DIRECT) &&
            isBackendAvailabilityErrno(errno)) {
            int e = errno;
            fallbackFromPreadDirect("open(O_DIRECT) failed for '" + path + "': " + std::string(std::strerror(e)), IoBackend::PreadBuffered);
            flags &= ~O_DIRECT;
            fd_ = openForBackend(path, flags, mode, /*hasMode=*/true);
        }
        if (fd_ < 0) {
            const char* modeName = (flags & O_DIRECT) ? "open(O_DIRECT)" : "open";
            throw std::runtime_error(std::string(modeName) + " failed for '" + path + "': " + std::strerror(errno));
        }

        if (usesIoUring()) {
            int fdArr[1] = {fd_};
            int responseCode = io_uring_register_files(&ring_, fdArr, 1);
            if (responseCode < 0) {
                int e = -responseCode;
                if (isAutoMode() && isBackendAvailabilityErrno(e)) {
                    fallbackFromIoUring("io_uring_register_files failed: " + std::string(std::strerror(e)), IoBackend::PreadDirect);
                } else {
                    close(fd_);
                    fd_ = -1;
                    throw std::runtime_error(std::string("io_uring_register_files failed: ") + std::strerror(e));
                }
            }
        }
        fileRegistered_ = true;
    }

    // Submit an async write using:
    // - a registered fixed buffer (bufIndex),
    // - the registered dump file as a fixed file,
    // - O_DIRECT semantics (alignment enforced here).
    //
    // Returns true if the SQE was queued successfully.
    // Returns false if the SQ ring is full (caller should drain completions and retry).
    //
    // Notes:
    // - Completion result will be available via CQE with cqe->user_data == userData.
    // - On completion, cqe->res is bytes written (>=0) or -errno (<0).
    bool submitWriteFixed(unsigned bufIndex, std::uint64_t fileOffsetBytes, std::uint32_t lenBytes, std::uint32_t bufOffsetBytes) {
        if (!buffersRegistered_)
            throw std::runtime_error("submitWriteFixed: no registered buffers");
        if (!fileRegistered_)
            throw std::runtime_error("submitWriteFixed: no registered file");
        if (bufIndex >= iovecs_.size())
            throw std::runtime_error("submitWriteFixed: bufIndex out of range");
        if (lenBytes == 0)
            throw std::runtime_error("submitWriteFixed: lenBytes is 0");

        // Fixed-buffer API keeps the direct-I/O alignment contract even if auto mode
        // had to fall back to buffered pread/pwrite for container compatibility.
        if ((fileOffsetBytes % kAlign) != 0)
            throw std::runtime_error("submitWriteFixed: fileOffsetBytes not 4k aligned");
        if ((uint64_t(lenBytes) % kAlign) != 0)
            throw std::runtime_error("submitWriteFixed: lenBytes not multiple of 4k");
        if ((uint64_t(bufOffsetBytes) % kAlign) != 0)
            throw std::runtime_error("submitWriteFixed: bufOffsetBytes not 4k aligned");

        if (uint64_t(bufOffsetBytes) + uint64_t(lenBytes) > iovecs_[bufIndex].iov_len)
            throw std::runtime_error("submitWriteFixed: (bufOffset+len) exceeds registered buffer length");

        // pointer INSIDE the registered buffer
        void* ptr = static_cast<uint8_t*>(iovecs_[bufIndex].iov_base) + bufOffsetBytes;

        if (!usesIoUring()) {
            return submitFallbackIo(FallbackOp::Write, ptr, lenBytes, fileOffsetBytes);
        }

        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe)
            return false;

        const std::uint64_t seq = nextSeq();
        ExactIoRequest req;
        req.op = ExactIoOp::Write;
        req.fixedBuffer = true;
        req.bufIndex = bufIndex;
        req.ptr = ptr;
        req.requestedBytes = lenBytes;
        req.remainingBytes = lenBytes;
        req.fileOffsetBytes = fileOffsetBytes;

        auto [it, inserted] = exactIoRequests_.emplace(seq, req);
        if (!inserted) {
            throw std::runtime_error("submitWriteFixed: duplicate exact-I/O sequence");
        }
        prepareExactIoSqe(sqe, seq, it->second);
        return true;
    }

    // Register a shard file for reading via O_DIRECT.
    // Replaces any previously registered file.
    void registerLoadFile(const std::string& path) { registerLoadFileImpl(path, /*useDirect=*/true); }

    // Register a file for cached reads. This intentionally does NOT use O_DIRECT: dataset
    // examples are good page-cache residents, and the batch loader wants the kernel cache
    // rather than bypassing it. The fd is still registered with io_uring so submissions can
    // use IOSQE_FIXED_FILE.
    void registerCachedLoadFile(const std::string& path) { registerLoadFileImpl(path, /*useDirect=*/false); }

    // Submit an async read into a registered fixed buffer. This is the
    // direct-I/O style read path and intentionally keeps the same 4k
    // offset/length constraints as O_DIRECT users. Cached dataset-example
    // reads with arbitrary byte counts should use submitReadCached().
    //
    // Returns false if SQ ring is full (caller should submit/drain and retry).
    bool submitReadFixed(unsigned bufIndex, std::uint64_t fileOffsetBytes, std::uint32_t lenBytes, std::uint32_t bufOffsetBytes) {
        if (!buffersRegistered_) {
            throw std::runtime_error("submitReadFixed: no registered buffers");
        }
        if (!fileRegistered_) {
            throw std::runtime_error("submitReadFixed: no registered file");
        }
        if (bufIndex >= iovecs_.size()) {
            throw std::runtime_error("submitReadFixed: bufIndex out of range");
        }
        if (lenBytes == 0) {
            throw std::runtime_error("submitReadFixed: lenBytes is 0");
        }

        // O_DIRECT alignment constraints (safe choice: 4k). Preserved for the
        // fixed-buffer API even when auto mode falls back to buffered pread.
        if ((fileOffsetBytes % kAlign) != 0) {
            throw std::runtime_error("submitReadFixed: fileOffsetBytes not 4k aligned");
        }
        if ((static_cast<std::uint64_t>(lenBytes) % kAlign) != 0) {
            throw std::runtime_error("submitReadFixed: lenBytes not multiple of 4k");
        }
        if ((static_cast<std::uint64_t>(bufOffsetBytes) % kAlign) != 0) {
            throw std::runtime_error("submitReadFixed: bufOffsetBytes not 4k aligned");
        }
        if (static_cast<std::size_t>(bufOffsetBytes) + static_cast<std::size_t>(lenBytes) > iovecs_[bufIndex].iov_len) {
            throw std::runtime_error("submitReadFixed: (bufOffset+len) exceeds registered buffer length");
        }

        void* ptr = static_cast<uint8_t*>(iovecs_[bufIndex].iov_base) + bufOffsetBytes;

        if (!usesIoUring()) {
            return submitFallbackIo(FallbackOp::Read, ptr, lenBytes, fileOffsetBytes);
        }

        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe)
            return false;

        const std::uint64_t seq = nextSeq();
        ExactIoRequest req;
        req.op = ExactIoOp::Read;
        req.fixedBuffer = true;
        req.bufIndex = bufIndex;
        req.ptr = ptr;
        req.requestedBytes = lenBytes;
        req.remainingBytes = lenBytes;
        req.fileOffsetBytes = fileOffsetBytes;

        auto [it, inserted] = exactIoRequests_.emplace(seq, req);
        if (!inserted) {
            throw std::runtime_error("submitReadFixed: duplicate exact-I/O sequence");
        }
        prepareExactIoSqe(sqe, seq, it->second);

        return true;  // caller batches and calls submit()
    }

    // Submit an async cached read into ordinary caller-owned memory.
    //
    // Unlike submitReadFixed(), this path does not require registered buffers and does not
    // impose O_DIRECT alignment constraints. The caller must keep `buf` alive until the
    // completion is delivered.
    bool submitReadCached(void* buf, std::uint64_t fileOffsetBytes, std::uint32_t lenBytes) {
        if (!fileRegistered_) {
            throw std::runtime_error("submitReadCached: no registered file");
        }
        if (buf == nullptr) {
            throw std::runtime_error("submitReadCached: null buffer");
        }
        if (lenBytes == 0) {
            throw std::runtime_error("submitReadCached: lenBytes is 0");
        }

        if (!usesIoUring()) {
            return submitFallbackIo(FallbackOp::Read, buf, lenBytes, fileOffsetBytes);
        }

        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe)
            return false;

        const std::uint64_t seq = nextSeq();
        ExactIoRequest req;
        req.op = ExactIoOp::Read;
        req.fixedBuffer = false;
        req.bufIndex = 0;
        req.ptr = buf;
        req.requestedBytes = lenBytes;
        req.remainingBytes = lenBytes;
        req.fileOffsetBytes = fileOffsetBytes;

        auto [it, inserted] = exactIoRequests_.emplace(seq, req);
        if (!inserted) {
            throw std::runtime_error("submitReadCached: duplicate exact-I/O sequence");
        }
        prepareExactIoSqe(sqe, seq, it->second);

        return true;
    }

    int submit() {
        if (!usesIoUring()) {
            std::lock_guard<std::mutex> guard(fallbackMutex_);
            int submitted = fallbackSubmittedSinceLastSubmit_;
            fallbackSubmittedSinceLastSubmit_ = 0;
            return submitted;
        }
        int responseCode = io_uring_submit(&ring_);
        if (responseCode < 0) {
            throw std::runtime_error(std::string("io_uring_submit failed: ") + std::strerror(-responseCode));
        }
        return responseCode;
    }

    struct Completion {
        std::uint64_t userData = 0;
        int responseCode = 0;  // >=0 bytes written/read, 0 for fsync, or -errno
    };

    Completion waitCompletionInOrder() {
        // Fast path: already have the next one buffered.
        auto it = pending_.find(nextDeliverSeq_);
        if (it != pending_.end()) {
            Completion out = it->second;
            pending_.erase(it);
            ++nextDeliverSeq_;
            return out;
        }

        // Otherwise, keep pulling CQEs until we see nextDeliverSeq_.
        while (true) {
            Completion c = waitCompletion();  // your existing primitive

            // If this is exactly the one we need, deliver it immediately.
            if (c.userData == nextDeliverSeq_) {
                ++nextDeliverSeq_;
                return c;
            }

            // Otherwise stash it (out-of-order completion).
            auto [insIt, inserted] = pending_.emplace(c.userData, c);
            if (!inserted) {
                // Duplicate userData indicates bug in caller's sequence or reuse while in flight.
                throw std::runtime_error("waitCompletionInOrder: duplicate completion userData=" + std::to_string(c.userData));
            }

            // Loop until we can deliver nextDeliverSeq_
            auto ready = pending_.find(nextDeliverSeq_);
            if (ready != pending_.end()) {
                Completion out = ready->second;
                pending_.erase(ready);
                ++nextDeliverSeq_;
                return out;
            }
        }
    }

    // Block until exactly targetCount completions are delivered in order.
    std::vector<Completion> waitCompletionsInOrder(std::size_t targetCount) {
        std::vector<Completion> out;
        out.reserve(targetCount);
        while (out.size() < targetCount) {
            Completion completion = waitCompletionInOrder();
            if (completion.responseCode <= 0) {
                if (completion.responseCode < 0) {
                    throw std::runtime_error(std::string("waitCompletionsInOrder: I/O completion failed: ") +
                                             std::strerror(-completion.responseCode));
                }
                throw std::runtime_error("waitCompletionsInOrder: read/write completion made no progress");
            }
            out.push_back(completion);
        }
        return out;
    }

    // Non-blocking: try to get the next completion in issue order.
    // Returns nullopt if the next-in-order completion isn't available yet.
    //
    // Semantics:
    // - May consume CQEs out of order and buffer them internally.
    // - Only returns a completion when userData == nextDeliverSeq_ (the head of line).
    std::optional<Completion> pollCompletionInOrder() {
        // Fast-path: already have the next one buffered.
        if (auto it = pending_.find(nextDeliverSeq_); it != pending_.end()) {
            Completion out = it->second;
            pending_.erase(it);
            ++nextDeliverSeq_;
            return out;
        }

        // Pull as many ready CQEs as are currently available (non-blocking)
        // and stash them in pending_.
        for (;;) {
            auto maybe = pollCompletion();
            if (!maybe.has_value()) {
                break;
            }

            Completion c = *maybe;

            // Stash; userData must be unique among in-flight ops.
            auto [it, inserted] = pending_.emplace(c.userData, c);
            if (!inserted) {
                throw std::runtime_error("pollCompletionInOrder: duplicate completion userData=" + std::to_string(c.userData));
            }

            // If we just received the head-of-line, we can stop early and return it.
            if (c.userData == nextDeliverSeq_) {
                break;
            }
        }

        // Try again: do we now have the head-of-line?
        if (auto it = pending_.find(nextDeliverSeq_); it != pending_.end()) {
            Completion out = it->second;
            pending_.erase(it);
            ++nextDeliverSeq_;
            return out;
        }

        return std::nullopt;
    }

    // Non-blocking: return up to maxCount completions that are complete *in order*.
    // Stops early if the next-in-order completion is not available yet (even if later ones are ready).
    std::vector<Completion> pollCompletionsInOrder(std::size_t maxCount = SIZE_MAX) {
        std::vector<Completion> out;
        if (maxCount == 0)
            return out;

        out.reserve(maxCount == SIZE_MAX ? 64 : maxCount);

        while (out.size() < maxCount) {
            auto one = pollCompletionInOrder();
            if (!one.has_value())
                break;
            out.push_back(*one);
        }
        return out;
    }

    // Returns the completion for the fsync op (res == 0 on success, -errno on failure).
    //
    // Contract: all prior read/write completions must have been delivered before
    // finishDumpedFile() is called. The fsync completion participates in the same
    // in-order sequence as reads/writes, so this method must not consume an
    // arbitrary CQE with waitCompletion(); doing so can steal a prior write
    // completion and corrupt nextDeliverSeq_.
    Completion finishDumpedFile(bool dataOnly = false) {
        if (!fileRegistered_ || fd_ < 0) {
            throw std::runtime_error("finishDumpedFile: no registered/open file");
        }
        if (hasUndeliveredCompletions()) {
            throw std::runtime_error("finishDumpedFile: pending read/write completions must be drained before fsync");
        }

        if (!usesIoUring()) {
            Completion c;
            c.userData = nextSeq();
            c.responseCode = dataOnly ? ::fdatasync(fd_) : ::fsync(fd_);
            if (c.responseCode != 0) {
                int e = errno;
                c.responseCode = -e;
                throw std::runtime_error(std::string("finishDumpedFile: fsync failed: ") + std::strerror(e));
            }
            ++nextDeliverSeq_;
            return c;
        }

        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            throw std::runtime_error("finishDumpedFile: SQ ring is full despite no pending completions");
        }

        unsigned fsyncFlags = dataOnly ? IORING_FSYNC_DATASYNC : 0;

        // Use fixed-file index 0 (since registerDumpFile registers exactly one fd).
        const std::uint64_t fsyncSeq = nextSeq();
        io_uring_prep_fsync(sqe, /*fd=*/0, fsyncFlags);
        sqe->flags |= IOSQE_FIXED_FILE;
        sqe->user_data = fsyncSeq;

        submit();

        Completion c = waitCompletionInOrder();
        if (c.userData != fsyncSeq) {
            throw std::runtime_error("finishDumpedFile: received non-fsync completion while waiting for fsync");
        }
        if (c.responseCode < 0) {
            throw std::runtime_error(std::string("finishDumpedFile: fsync failed: ") + std::strerror(-c.responseCode));
        }

        return c;
    }

    // Accessors for later stages (submission/completion code).
    io_uring* ring() {
        if (!usesIoUring()) {
            throw std::runtime_error(std::string("UringDirect::ring requested while active I/O backend is ") + activeBackendName());
        }
        return &ring_;
    }
    int fd() const { return fd_; }
    bool buffersRegistered() const { return buffersRegistered_; }
    bool fileRegistered() const { return fileRegistered_; }

   protected:
    // Non-blocking: try to get one completion.
    // Returns std::nullopt if none are available.
    std::optional<Completion> pollCompletion() {
        if (!usesIoUring()) {
            return pollFallbackCompletion();
        }

        io_uring_cqe* cqe = nullptr;
        int responseCode = io_uring_peek_cqe(&ring_, &cqe);
        if (responseCode == -EAGAIN || cqe == nullptr) {
            return std::nullopt;
        }
        if (responseCode < 0) {
            throw std::runtime_error(std::string("io_uring_peek_cqe failed: ") + std::strerror(-responseCode));
        }

        Completion out;
        out.userData = cqe->user_data;
        out.responseCode = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        return normalizeIoUringCompletion(out);
    }

    // Blocking: wait until a logical completion is available, then return it.
    Completion waitCompletion() {
        if (!usesIoUring()) {
            return waitFallbackCompletion();
        }

        for (;;) {
            io_uring_cqe* cqe = nullptr;
            int responseCode = io_uring_wait_cqe(&ring_, &cqe);
            if (responseCode < 0) {
                throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::strerror(-responseCode));
            }

            Completion out;
            out.userData = cqe->user_data;
            out.responseCode = cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
            std::optional<Completion> normalized = normalizeIoUringCompletion(out);
            if (normalized.has_value()) {
                return *normalized;
            }
        }
    }

    // Drain up to maxCount completions (non-blocking).
    // Returns the completions collected.
    std::vector<Completion> pollCompletions(std::size_t maxCount = SIZE_MAX) {
        std::vector<Completion> out;
        if (maxCount == 0)
            return out;

        out.reserve(maxCount == SIZE_MAX ? 64 : maxCount);

        while (out.size() < maxCount) {
            auto one = pollCompletion();
            if (!one.has_value())
                break;
            out.push_back(*one);
        }
        return out;
    }

    // Block until exactly targetCount completions have been collected.
    // This is useful for backpressure: wait until you free enough in-flight slots.
    std::vector<Completion> waitCompletions(std::size_t targetCount) {
        if (targetCount == 0)
            return {};

        std::vector<Completion> out;
        out.reserve(targetCount);

        while (out.size() < targetCount) {
            out.push_back(waitCompletion());
        }
        return out;
    }

   private:
    enum class ExactIoOp { Read, Write };

    struct ExactIoRequest {
        ExactIoOp op = ExactIoOp::Read;
        bool fixedBuffer = false;
        unsigned bufIndex = 0;
        void* ptr = nullptr;
        std::uint32_t requestedBytes = 0;
        std::uint32_t remainingBytes = 0;
        std::uint32_t submittedBytes = 0;
        std::uint64_t fileOffsetBytes = 0;
    };

    void prepareExactIoSqe(io_uring_sqe* sqe, std::uint64_t seq, ExactIoRequest& req) {
        std::uint32_t bytesToSubmit = req.remainingBytes;
#ifdef THOR_GTEST
        bytesToSubmit = limitNextIoUringSubmissionBytesForTest(bytesToSubmit);
#endif
        if (bytesToSubmit == 0 || bytesToSubmit > req.remainingBytes) {
            throw std::runtime_error("prepareExactIoSqe: invalid exact-I/O submission byte count");
        }
        req.submittedBytes = bytesToSubmit;

        if (req.op == ExactIoOp::Write) {
            if (!req.fixedBuffer) {
                throw std::runtime_error("prepareExactIoSqe: cached writes are not supported");
            }
            io_uring_prep_write_fixed(sqe,
                                      /*fd=*/0,
                                      /*buf=*/req.ptr,
                                      /*nbytes=*/bytesToSubmit,
                                      /*offset=*/static_cast<off_t>(req.fileOffsetBytes),
                                      /*buf_index=*/req.bufIndex);
        } else {
            if (req.fixedBuffer) {
                io_uring_prep_read_fixed(sqe,
                                         /*fd=*/0,
                                         /*buf=*/req.ptr,
                                         /*nbytes=*/bytesToSubmit,
                                         /*offset=*/static_cast<off_t>(req.fileOffsetBytes),
                                         /*buf_index=*/req.bufIndex);
            } else {
                io_uring_prep_read(sqe,
                                   /*fd=*/0,
                                   /*buf=*/req.ptr,
                                   /*nbytes=*/bytesToSubmit,
                                   /*offset=*/static_cast<off_t>(req.fileOffsetBytes));
            }
        }
        sqe->flags |= IOSQE_FIXED_FILE;
        sqe->user_data = seq;
    }

    void submitExactIoContinuation(std::uint64_t seq, ExactIoRequest& req) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            submit();
            sqe = io_uring_get_sqe(&ring_);
        }
        if (!sqe) {
            throw std::runtime_error("UringDirect: unable to queue exact-I/O continuation after short completion");
        }

        prepareExactIoSqe(sqe, seq, req);
        submit();
    }

    std::optional<Completion> normalizeIoUringCompletion(Completion completion) {
        auto it = exactIoRequests_.find(completion.userData);
        if (it == exactIoRequests_.end()) {
            return completion;
        }

        ExactIoRequest& req = it->second;
        if (completion.responseCode < 0) {
            const int e = -completion.responseCode;
            exactIoRequests_.erase(it);
            throw std::runtime_error(std::string("UringDirect: exact io_uring request failed: ") + std::strerror(e));
        }
        if (completion.responseCode == 0) {
            const std::uint32_t remaining = req.remainingBytes;
            exactIoRequests_.erase(it);
            throw std::runtime_error("UringDirect: io_uring made no progress with " + std::to_string(remaining) +
                                     " bytes remaining for exact-I/O request");
        }

        const auto transferred = static_cast<std::uint32_t>(completion.responseCode);
        if (transferred > req.submittedBytes) {
            const std::uint32_t submitted = req.submittedBytes;
            exactIoRequests_.erase(it);
            throw std::runtime_error("UringDirect: io_uring completion responseCode " + std::to_string(completion.responseCode) +
                                     " exceeds submitted exact-I/O byte count " + std::to_string(submitted));
        }

        req.ptr = static_cast<std::uint8_t*>(req.ptr) + transferred;
        req.fileOffsetBytes += transferred;
        req.remainingBytes -= transferred;

        if (req.remainingBytes == 0) {
            Completion out = completion;
            out.responseCode = static_cast<int>(req.requestedBytes);
            exactIoRequests_.erase(it);
            return out;
        }

        submitExactIoContinuation(completion.userData, req);
        return std::nullopt;
    }

    void initializeBackend(unsigned queueDepth) {
        if (requestedBackend_ == IoBackend::PreadBuffered) {
            activeBackend_ = IoBackend::PreadBuffered;
            return;
        }
        if (requestedBackend_ == IoBackend::PreadDirect) {
            activeBackend_ = IoBackend::PreadDirect;
            return;
        }

        int responseCode = ioUringQueueInitForInstance(queueDepth);
        if (responseCode >= 0) {
            ringInited_ = true;
            activeBackend_ = IoBackend::UringDirect;
            return;
        }

        int e = -responseCode;
        if (requestedBackend_ == IoBackend::UringDirect) {
            throw std::runtime_error(std::string("io_uring_queue_init failed: ") + std::strerror(e));
        }

        activeBackend_ = IoBackend::PreadDirect;
        emitCompatibilityWarningOnce("io_uring_queue_init failed: " + std::string(std::strerror(e)), activeBackend_);
    }

    void registerLoadFileImpl(const std::string& path, bool useDirect) {
        closeRegisteredFile();

        bool requestDirect = useDirect && activeBackend_ != IoBackend::PreadBuffered;
        int flags = O_RDONLY | O_CLOEXEC;
        if (requestDirect)
            flags |= O_DIRECT;

        fd_ = openForBackend(path, flags, 0, /*hasMode=*/false);
        if (fd_ < 0 && isAutoMode() && activeBackend_ == IoBackend::UringDirect && useDirect && (flags & O_DIRECT) &&
            isBackendAvailabilityErrno(errno)) {
            int e = errno;
            fallbackFromIoUring("open(O_DIRECT, RDONLY) failed for '" + path + "': " + std::string(std::strerror(e)),
                                IoBackend::PreadBuffered);
            requestDirect = false;
            flags &= ~O_DIRECT;
            fd_ = openForBackend(path, flags, 0, /*hasMode=*/false);
        }
        if (fd_ < 0 && isAutoMode() && activeBackend_ == IoBackend::PreadDirect && useDirect && (flags & O_DIRECT) &&
            isBackendAvailabilityErrno(errno)) {
            int e = errno;
            fallbackFromPreadDirect("open(O_DIRECT, RDONLY) failed for '" + path + "': " + std::string(std::strerror(e)),
                                    IoBackend::PreadBuffered);
            requestDirect = false;
            flags &= ~O_DIRECT;
            fd_ = openForBackend(path, flags, 0, /*hasMode=*/false);
        }
        if (fd_ < 0) {
            const char* mode = requestDirect ? "open(O_DIRECT, RDONLY)" : "open(RDONLY)";
            throw std::runtime_error(std::string(mode) + " failed for '" + path + "': " + std::strerror(errno));
        }

        // Optional sanity: regular file
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            int e = errno;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("fstat failed for '" + path + "': " + std::strerror(e));
        }
        if (!S_ISREG(st.st_mode)) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("registerReadFile: path is not a regular file: '" + path + "'");
        }

        if (usesIoUring()) {
            // Register as fixed file index 0
            int fdArr[1] = {fd_};
            int rc = io_uring_register_files(&ring_, fdArr, 1);
            if (rc < 0) {
                int e = -rc;
                if (isAutoMode() && isBackendAvailabilityErrno(e)) {
                    fallbackFromIoUring("io_uring_register_files failed: " + std::string(std::strerror(e)), IoBackend::PreadDirect);
                } else {
                    close(fd_);
                    fd_ = -1;
                    throw std::runtime_error(std::string("io_uring_register_files failed: ") + std::strerror(e));
                }
            }
        }
        fileRegistered_ = true;
    }

    static bool isAligned(const void* p, std::size_t align) {
        auto v = reinterpret_cast<std::uintptr_t>(p);
        return (v % align) == 0;
    }

    void closeRegisteredFile() {
        if (fileRegistered_ && hasUndeliveredCompletions()) {
            throw std::runtime_error(std::string("closeRegisteredFile: pending ") + activeBackendName() +
                                     " completions must be drained before replacing the file");
        }
        if (fileRegistered_ && usesIoUring()) {
            int rc = io_uring_unregister_files(&ring_);
            if (rc < 0) {
                throw std::runtime_error(std::string("io_uring_unregister_files failed: ") + std::strerror(-rc));
            }
        }
        fileRegistered_ = false;
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void fallbackFromIoUring(const std::string& reason, IoBackend fallbackBackend) {
        // Keep any already-open fd logically registered for the fallback backend, but
        // tear down kernel io_uring registrations before exiting the ring.
        if (ringInited_) {
            if (fileRegistered_) {
                (void)io_uring_unregister_files(&ring_);
            }
            if (buffersRegistered_) {
                (void)io_uring_unregister_buffers(&ring_);
            }
            io_uring_queue_exit(&ring_);
            ringInited_ = false;
        }
        activeBackend_ = fallbackBackend;
        emitCompatibilityWarningOnce(reason, fallbackBackend);
    }

    void fallbackFromPreadDirect(const std::string& reason, IoBackend fallbackBackend) {
        activeBackend_ = fallbackBackend;
        emitCompatibilityWarningOnce(reason, fallbackBackend);
    }

    bool usesIoUring() const { return activeBackend_ == IoBackend::UringDirect; }
    bool isAutoMode() const { return requestedBackend_ == IoBackend::Auto; }
    bool hasUndeliveredCompletions() const {
        return nextIssueSeq_ != nextDeliverSeq_ || !pending_.empty() || !exactIoRequests_.empty() || hasFallbackCompletionsReady();
    }

    static const char* backendName(IoBackend backend) {
        switch (backend) {
            case IoBackend::Auto:
                return "auto";
            case IoBackend::UringDirect:
                return "uring_direct";
            case IoBackend::PreadDirect:
                return "pread_direct";
            case IoBackend::PreadBuffered:
                return "pread_buffered";
        }
        return "unknown";
    }

    static bool isBackendAvailabilityErrno(int e) {
        return e == EPERM || e == EACCES || e == ENOSYS || e == EINVAL || e == EOPNOTSUPP
#ifdef ENOTSUP
               || e == ENOTSUP
#endif
            ;
    }

    static void emitCompatibilityWarningOnce(const std::string& reason, IoBackend fallbackBackend) {
        std::lock_guard<std::mutex> guard(compatibilityWarningMutex());
        if (compatibilityWarningEmitted()) {
            return;
        }
        compatibilityWarningEmitted() = true;
        std::cerr << "Thor warning: I/O backend uring_direct is unavailable or not usable in this runtime.\n"
                  << "  Reason: " << reason << "\n"
                  << "  Falling back to " << backendName(fallbackBackend) << ".\n"
                  << "  Backend order for THOR_IO_BACKEND=auto is: uring_direct, pread_direct, pread_buffered.\n"
                  << "  Docker/dev-container workaround: run with a seccomp profile that allows io_uring_setup,\n"
                  << "    io_uring_enter, and io_uring_register, or use --security-opt seccomp=unconfined when that is acceptable.\n"
                  << "  Managed cloud training environments may block io_uring through container seccomp,\n"
                  << "    kernel.io_uring_disabled, or provider security policy; those settings are often not configurable by jobs.\n"
                  << "  For deterministic behavior set THOR_IO_BACKEND=uring_direct, pread_direct, or pread_buffered.\n";
    }

    static std::mutex& compatibilityWarningMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static bool& compatibilityWarningEmitted() {
        static bool emitted = false;
        return emitted;
    }

    static int openForBackend(const std::string& path, int flags, mode_t mode, bool hasMode) {
#ifdef THOR_GTEST
        if ((flags & O_DIRECT) && testDirectOpenUnavailable()) {
            errno = EINVAL;
            return -1;
        }
#endif
        if (hasMode) {
            return ::open(path.c_str(), flags, mode);
        }
        return ::open(path.c_str(), flags);
    }

    int initIoUringRing(unsigned queueDepth) {
#ifdef THOR_GTEST
        if (testIoUringQueueInitResult().has_value()) {
            return *testIoUringQueueInitResult();
        }
#endif
        return io_uring_queue_init(queueDepth, &ring_, /*flags=*/0);
    }

    // initializeBackend needs to initialize this instance's ring, not a temporary.
    // Keep the test hook centralized by routing through initIoUringRing().
    int ioUringQueueInitForInstance(unsigned queueDepth) { return initIoUringRing(queueDepth); }

    enum class FallbackOp { Read, Write };

    struct FallbackRequest {
        FallbackOp op = FallbackOp::Read;
        Completion completion;
        int fd = -1;
        void* ptr = nullptr;
        std::uint32_t lenBytes = 0;
        std::uint64_t fileOffsetBytes = 0;
    };

    void ensureFallbackWorkers() {
        if (usesIoUring()) {
            return;
        }

        std::lock_guard<std::mutex> guard(fallbackMutex_);
        if (!fallbackWorkers_.empty()) {
            return;
        }
        fallbackStop_ = false;

        const std::size_t numWorkers = fallbackWorkerCount();
        fallbackWorkers_.reserve(numWorkers);
        for (std::size_t i = 0; i < numWorkers; ++i) {
            fallbackWorkers_.emplace_back([this] { fallbackWorkerLoop(); });
        }
    }

    std::size_t fallbackWorkerCount() const {
        return std::max<std::size_t>(1, std::min<std::size_t>(fallbackQueueDepth_, 64));
    }

    bool submitFallbackIo(FallbackOp op, void* ptr, std::uint32_t lenBytes, std::uint64_t fileOffsetBytes) {
        ensureFallbackWorkers();

        std::lock_guard<std::mutex> guard(fallbackMutex_);
        if (fallbackStop_) {
            throw std::runtime_error("submitFallbackIo: fallback worker pool is shutting down");
        }
        if (fallbackInFlight_ >= fallbackQueueDepth_) {
            return false;
        }

        FallbackRequest req;
        req.op = op;
        req.completion.userData = nextSeq();
        req.fd = fd_;
        req.ptr = ptr;
        req.lenBytes = lenBytes;
        req.fileOffsetBytes = fileOffsetBytes;
        fallbackWork_.push_back(req);
        ++fallbackInFlight_;
        ++fallbackSubmittedSinceLastSubmit_;
        fallbackWorkCv_.notify_one();
        return true;
    }

    std::optional<Completion> pollFallbackCompletion() {
        std::lock_guard<std::mutex> guard(fallbackMutex_);
        if (fallbackCompleted_.empty()) {
            return std::nullopt;
        }
        Completion out = fallbackCompleted_.front();
        fallbackCompleted_.pop_front();
        if (fallbackInFlight_ > 0) {
            --fallbackInFlight_;
        }
        return out;
    }

    Completion waitFallbackCompletion() {
        std::unique_lock<std::mutex> lock(fallbackMutex_);
        fallbackCompletedCv_.wait(lock, [this] { return !fallbackCompleted_.empty() || (fallbackStop_ && fallbackInFlight_ == 0); });
        if (fallbackCompleted_.empty()) {
            throw std::runtime_error("waitCompletion: no pending completion for pread/pwrite backend");
        }
        Completion out = fallbackCompleted_.front();
        fallbackCompleted_.pop_front();
        if (fallbackInFlight_ > 0) {
            --fallbackInFlight_;
        }
        return out;
    }

    bool hasFallbackCompletionsReady() const {
        std::lock_guard<std::mutex> guard(fallbackMutex_);
        return !fallbackCompleted_.empty() || !fallbackWork_.empty() || fallbackInFlight_ != 0;
    }

    void fallbackWorkerLoop() {
#ifdef THOR_GTEST
        testCurrentThreadIsFallbackWorker() = true;
#endif
        for (;;) {
            FallbackRequest req;
            {
                std::unique_lock<std::mutex> lock(fallbackMutex_);
                fallbackWorkCv_.wait(lock, [this] { return fallbackStop_ || !fallbackWork_.empty(); });
                if (fallbackStop_ && fallbackWork_.empty()) {
                    break;
                }
                req = fallbackWork_.front();
                fallbackWork_.pop_front();
            }

#ifdef THOR_GTEST
            maybeBlockFallbackWorkerForTest();
#endif

            if (req.op == FallbackOp::Read) {
                req.completion.responseCode = preadAll(req.fd, req.ptr, req.lenBytes, req.fileOffsetBytes);
            } else {
                req.completion.responseCode = pwriteAll(req.fd, req.ptr, req.lenBytes, req.fileOffsetBytes);
            }

            {
                std::lock_guard<std::mutex> lock(fallbackMutex_);
                fallbackCompleted_.push_back(req.completion);
            }
            fallbackCompletedCv_.notify_one();
        }
#ifdef THOR_GTEST
        testCurrentThreadIsFallbackWorker() = false;
#endif
    }

    void shutdownFallbackWorkersNoexcept() noexcept {
        {
            std::lock_guard<std::mutex> guard(fallbackMutex_);
            fallbackStop_ = true;
        }
        fallbackWorkCv_.notify_all();
        for (std::thread& worker : fallbackWorkers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        fallbackWorkers_.clear();
    }

    static int preadAll(int fd, void* buf, std::uint32_t lenBytes, std::uint64_t fileOffsetBytes) {
        std::uint64_t done = 0;
        while (done < lenBytes) {
            ssize_t n = ::pread(fd,
                                static_cast<std::uint8_t*>(buf) + done,
                                static_cast<std::size_t>(lenBytes - done),
                                static_cast<off_t>(fileOffsetBytes + done));
            if (n < 0) {
                return -errno;
            }
            if (n == 0) {
                return -EIO;
            }
            done += static_cast<std::uint64_t>(n);
        }
        return static_cast<int>(done);
    }

    static int pwriteAll(int fd, const void* buf, std::uint32_t lenBytes, std::uint64_t fileOffsetBytes) {
        std::uint64_t done = 0;
        while (done < lenBytes) {
            ssize_t n = ::pwrite(fd,
                                 static_cast<const std::uint8_t*>(buf) + done,
                                 static_cast<std::size_t>(lenBytes - done),
                                 static_cast<off_t>(fileOffsetBytes + done));
            if (n < 0) {
                return -errno;
            }
            if (n == 0) {
                return -EIO;
            }
            done += static_cast<std::uint64_t>(n);
        }
        return static_cast<int>(done);
    }

#ifdef THOR_GTEST
    static std::optional<int>& testIoUringQueueInitResult() {
        static std::optional<int> responseCode;
        return responseCode;
    }

    static std::mutex& testIoUringSubmissionByteLimitMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::optional<std::uint32_t>& testNextIoUringSubmissionByteLimit() {
        static std::optional<std::uint32_t> limitBytes;
        return limitBytes;
    }

    static std::uint32_t limitNextIoUringSubmissionBytesForTest(std::uint32_t requestedBytes) {
        std::lock_guard<std::mutex> guard(testIoUringSubmissionByteLimitMutex());
        if (!testNextIoUringSubmissionByteLimit().has_value()) {
            return requestedBytes;
        }
        const std::uint32_t limitBytes = *testNextIoUringSubmissionByteLimit();
        if (requestedBytes <= limitBytes) {
            return requestedBytes;
        }
        testNextIoUringSubmissionByteLimit().reset();
        return limitBytes;
    }

    static bool& testDirectOpenUnavailable() {
        static bool unavailable = false;
        return unavailable;
    }

    static std::mutex& testFallbackWorkerBlockMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::condition_variable& testFallbackWorkerBlockCv() {
        static std::condition_variable cv;
        return cv;
    }

    static bool& testFallbackWorkerBlockEnabled() {
        static bool enabled = false;
        return enabled;
    }

    static std::size_t& testFallbackWorkerStartedCountValue() {
        static std::size_t count = 0;
        return count;
    }

    static bool& testCurrentThreadIsFallbackWorker() {
        thread_local bool isWorker = false;
        return isWorker;
    }

    static void maybeBlockFallbackWorkerForTest() {
        if (!testCurrentThreadIsFallbackWorker()) {
            return;
        }
        std::unique_lock<std::mutex> lock(testFallbackWorkerBlockMutex());
        if (!testFallbackWorkerBlockEnabled()) {
            return;
        }
        ++testFallbackWorkerStartedCountValue();
        testFallbackWorkerBlockCv().notify_all();
        testFallbackWorkerBlockCv().wait(lock, [] { return !testFallbackWorkerBlockEnabled(); });
    }
#endif

    void moveFrom(UringDirect&& other) noexcept {
        other.shutdownFallbackWorkersNoexcept();

        ring_ = other.ring_;
        ringInited_ = other.ringInited_;
        other.ringInited_ = false;

        requestedBackend_ = other.requestedBackend_;
        activeBackend_ = other.activeBackend_;

        fd_ = other.fd_;
        other.fd_ = -1;

        iovecs_ = std::move(other.iovecs_);

        buffersRegistered_ = other.buffersRegistered_;
        other.buffersRegistered_ = false;

        fileRegistered_ = other.fileRegistered_;
        other.fileRegistered_ = false;

        pending_ = std::move(other.pending_);
        exactIoRequests_ = std::move(other.exactIoRequests_);
        {
            std::lock_guard<std::mutex> lock(other.fallbackMutex_);
            fallbackWork_ = std::move(other.fallbackWork_);
            fallbackCompleted_ = std::move(other.fallbackCompleted_);
            fallbackInFlight_ = other.fallbackInFlight_;
            fallbackSubmittedSinceLastSubmit_ = other.fallbackSubmittedSinceLastSubmit_;
            other.fallbackInFlight_ = 0;
            other.fallbackSubmittedSinceLastSubmit_ = 0;
        }
        fallbackQueueDepth_ = other.fallbackQueueDepth_;
        fallbackStop_ = false;

        nextIssueSeq_ = other.nextIssueSeq_;
        nextDeliverSeq_ = other.nextDeliverSeq_;
    }

    uint64_t nextSeq() { return nextIssueSeq_++; }

    io_uring ring_{};
    bool ringInited_ = false;

    IoBackend requestedBackend_ = IoBackend::Auto;
    IoBackend activeBackend_ = IoBackend::PreadBuffered;

    int fd_ = -1;

    std::vector<iovec> iovecs_;
    bool buffersRegistered_ = false;
    bool fileRegistered_ = false;

    std::unordered_map<uint64_t, Completion> pending_;
    std::unordered_map<uint64_t, ExactIoRequest> exactIoRequests_;

    mutable std::mutex fallbackMutex_;
    std::condition_variable fallbackWorkCv_;
    std::condition_variable fallbackCompletedCv_;
    std::deque<FallbackRequest> fallbackWork_;
    std::deque<Completion> fallbackCompleted_;
    std::vector<std::thread> fallbackWorkers_;
    bool fallbackStop_ = false;
    std::size_t fallbackQueueDepth_ = 64;
    std::size_t fallbackInFlight_ = 0;
    int fallbackSubmittedSinceLastSubmit_ = 0;

    static constexpr uint64_t SEQUENCE_START = 1000;
    uint64_t nextIssueSeq_ = SEQUENCE_START;
    uint64_t nextDeliverSeq_ = SEQUENCE_START;
};
