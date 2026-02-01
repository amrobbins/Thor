#pragma once

#include <liburing.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

class UringDirect {
   public:
    static constexpr std::size_t kAlign = 4096;

    explicit UringDirect(unsigned queueDepth = 64) {
        int responseCode = io_uring_queue_init(queueDepth, &ring_, /*flags=*/0);
        if (responseCode < 0) {
            throw std::runtime_error(std::string("io_uring_queue_init failed: ") + std::strerror(-responseCode));
        }
        ringInited_ = true;
    }

    ~UringDirect() {
        // Unregister in reverse order (best-effort).
        if (fileRegistered_) {
            (void)io_uring_unregister_files(&ring_);
            fileRegistered_ = false;
        }
        if (buffersRegistered_) {
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
            int rc = io_uring_unregister_buffers(&ring_);
            if (rc < 0) {
                throw std::runtime_error(std::string("io_uring_unregister_buffers failed: ") + std::strerror(-rc));
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

        int rc = io_uring_register_buffers(&ring_, iovecs_.data(), static_cast<unsigned>(iovecs_.size()));
        if (rc < 0) {
            throw std::runtime_error(std::string("io_uring_register_buffers failed: ") + std::strerror(-rc));
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
        // Close/unregister any previous file.
        if (fileRegistered_) {
            int responseCode = io_uring_unregister_files(&ring_);
            if (responseCode < 0) {
                throw std::runtime_error(std::string("io_uring_unregister_files failed: ") + std::strerror(-responseCode));
            }
            fileRegistered_ = false;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }

        int flags = O_WRONLY | O_CREAT | O_CLOEXEC | O_DIRECT;
        if (truncate)
            flags |= O_TRUNC;

        fd_ = open(path.c_str(), flags, mode);
        if (fd_ < 0) {
            throw std::runtime_error("open(O_DIRECT) failed for '" + path + "': " + std::strerror(errno));
        }

        // Register file with io_uring for "fixed file" ops.
        int fdArr[1] = {fd_};
        int responseCode = io_uring_register_files(&ring_, fdArr, 1);
        if (responseCode < 0) {
            int e = -responseCode;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("io_uring_register_files failed: ") + std::strerror(e));
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

        // O_DIRECT alignment
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

        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe)
            return false;

        io_uring_prep_write_fixed(sqe,
                                  /*fd=*/0,
                                  /*buf=*/ptr,
                                  /*nbytes=*/lenBytes,
                                  /*offset=*/static_cast<off_t>(fileOffsetBytes),
                                  /*buf_index=*/bufIndex);

        sqe->flags |= IOSQE_FIXED_FILE;
        sqe->user_data = nextSeq();
        return true;
    }

    // // Submit an async write using:
    // // - a registered fixed buffer (bufIndex),
    // // - the registered dump file as a fixed file,
    // // - O_DIRECT semantics (alignment enforced here).
    // //
    // // Returns true if the SQE was queued successfully.
    // // Returns false if the SQ ring is full (caller should drain completions and retry).
    // //
    // // Notes:
    // // - Completion result will be available via CQE with cqe->user_data == userData.
    // // - On completion, cqe->res is bytes written (>=0) or -errno (<0).
    // bool submitWriteFixed(unsigned bufIndex, std::uint64_t fileOffsetBytes, std::uint32_t lenBytes, ) {
    //     if (!buffersRegistered_) {
    //         throw std::runtime_error("submitWriteFixed: no registered buffers");
    //     }
    //     if (!fileRegistered_) {
    //         throw std::runtime_error("submitWriteFixed: no registered file");
    //     }
    //     if (bufIndex >= iovecs_.size()) {
    //         throw std::runtime_error("submitWriteFixed: bufIndex out of range");
    //     }
    //     if (lenBytes == 0) {
    //         throw std::runtime_error("submitWriteFixed: lenBytes is 0");
    //     }
    //
    //     // O_DIRECT alignment constraints (common safe choice: 4k).
    //     if ((fileOffsetBytes % kAlign) != 0) {
    //         throw std::runtime_error("submitWriteFixed: fileOffsetBytes not 4k aligned");
    //     }
    //     if ((static_cast<std::uint64_t>(lenBytes) % kAlign) != 0) {
    //         throw std::runtime_error("submitWriteFixed: lenBytes not multiple of 4k");
    //     }
    //     if (static_cast<std::size_t>(lenBytes) > iovecs_[bufIndex].iov_len) {
    //         throw std::runtime_error("submitWriteFixed: lenBytes exceeds registered buffer length");
    //     }
    //
    //     io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    //     if (!sqe) {
    //         return false;  // SQ full; caller should submit/drain and retry
    //     }
    //
    //     // Use fixed buffer + fixed file.
    //     //
    //     // Important: io_uring_prep_write_fixed takes:
    //     //   (sqe, fd, buf, nbytes, offset, buf_index)
    //     //
    //     // For fixed file, set IOSQE_FIXED_FILE and use fd=0 (index into registered files).
    //     io_uring_prep_write_fixed(sqe,
    //                               /*fd=*/0,  // fixed file index 0
    //                               /*buf=*/iovecs_[bufIndex].iov_base,
    //                               /*nbytes=*/lenBytes,
    //                               /*offset=*/static_cast<off_t>(fileOffsetBytes),
    //                               /*buf_index=*/bufIndex);
    //     sqe->flags |= IOSQE_FIXED_FILE;
    //     sqe->user_data = nextSeq();
    //
    //     // We do NOT call io_uring_submit() here so the caller can batch.
    //     // Caller should call submit() / flush()
    //     return true;
    // }

    // Register a shard file for reading via O_DIRECT.
    // Replaces any previously registered file.
    void registerLoadFile(const std::string& path) {
        // Unregister/close previous
        if (fileRegistered_) {
            int rc = io_uring_unregister_files(&ring_);
            if (rc < 0) {
                throw std::runtime_error(std::string("io_uring_unregister_files failed: ") + std::strerror(-rc));
            }
            fileRegistered_ = false;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }

        int flags = O_RDONLY | O_CLOEXEC | O_DIRECT;
        fd_ = open(path.c_str(), flags);
        if (fd_ < 0) {
            throw std::runtime_error("open(O_DIRECT, RDONLY) failed for '" + path + "': " + std::strerror(errno));
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

        // Register as fixed file index 0
        int fdArr[1] = {fd_};
        int rc = io_uring_register_files(&ring_, fdArr, 1);
        if (rc < 0) {
            int e = -rc;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("io_uring_register_files failed: ") + std::strerror(e));
        }
        fileRegistered_ = true;
    }

    // Submit an async read into a registered fixed buffer.
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

        // O_DIRECT alignment constraints (safe choice: 4k)
        if ((fileOffsetBytes % kAlign) != 0) {
            throw std::runtime_error("submitReadFixed: fileOffsetBytes not 4k aligned");
        }
        if ((static_cast<std::uint64_t>(lenBytes) % kAlign) != 0) {
            throw std::runtime_error("submitReadFixed: lenBytes not multiple of 4k");
        }
        if (static_cast<std::size_t>(lenBytes) > iovecs_[bufIndex].iov_len) {
            throw std::runtime_error("submitReadFixed: lenBytes exceeds registered buffer length");
        }

        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe)
            return false;

        // Fixed file index 0 + fixed buffer index bufIndex with offset.
        void* ptr = static_cast<uint8_t*>(iovecs_[bufIndex].iov_base) + bufOffsetBytes;
        io_uring_prep_read_fixed(sqe,
                                 /*fd=*/0,  // fixed-file index 0
                                 /*buf=*/ptr,
                                 /*nbytes=*/lenBytes,
                                 /*offset=*/static_cast<off_t>(fileOffsetBytes),
                                 /*buf_index=*/bufIndex);
        sqe->flags |= IOSQE_FIXED_FILE;
        sqe->user_data = nextSeq();

        return true;  // caller batches and calls submit()
    }

    int submit() {
        int responseCode = io_uring_submit(&ring_);
        if (responseCode < 0) {
            throw std::runtime_error(std::string("io_uring_submit failed: ") + std::strerror(-responseCode));
        }
        return responseCode;
    }

    struct Completion {
        std::uint64_t userData = 0;
        int responseCode = 0;  // >=0 bytes written, or -errno
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
            out.push_back(waitCompletionInOrder());
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
            io_uring_cqe* cqe = nullptr;
            int rc = io_uring_peek_cqe(&ring_, &cqe);

            if (rc == -EAGAIN || cqe == nullptr) {
                break;  // no more CQEs ready right now
            }
            if (rc < 0) {
                throw std::runtime_error(std::string("io_uring_peek_cqe failed: ") + std::strerror(-rc));
            }

            Completion c;
            c.userData = cqe->user_data;
            c.responseCode = cqe->res;

            io_uring_cqe_seen(&ring_, cqe);

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
    Completion finishDumpedFile(bool dataOnly = false) {
        if (!fileRegistered_ || fd_ < 0) {
            throw std::runtime_error("finishDumpedFile: no registered/open file");
        }

        // Get an SQE; if full, submit and wait for one completion, then retry once.
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            // push any pending SQEs
            submit();
            // wait for any completion to free CQ/SQ pressure
            (void)waitCompletion();
            sqe = io_uring_get_sqe(&ring_);
            if (!sqe) {
                throw std::runtime_error("finishDumpedFile: SQ ring still full");
            }
        }

        unsigned fsyncFlags = dataOnly ? IORING_FSYNC_DATASYNC : 0;

        // Use fixed-file index 0 (since registerDumpFile registers exactly one fd)
        io_uring_prep_fsync(sqe, /*fd=*/0, fsyncFlags);
        sqe->flags |= IOSQE_FIXED_FILE;
        sqe->user_data = nextSeq();

        submit();

        Completion c = waitCompletion();
        if (c.responseCode < 0) {
            throw std::runtime_error(std::string("finishDumpedFile: fsync failed: ") + std::strerror(-c.responseCode));
        }
        ++nextDeliverSeq_;

        return c;
    }

    // Accessors for later stages (submission/completion code).
    io_uring* ring() { return &ring_; }
    int fd() const { return fd_; }
    bool buffersRegistered() const { return buffersRegistered_; }
    bool fileRegistered() const { return fileRegistered_; }

   protected:
    // Non-blocking: try to get one completion.
    // Returns std::nullopt if none are available.
    std::optional<Completion> pollCompletion() {
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
        return out;
    }

    // Blocking: wait until at least one completion is available, then return it.
    Completion waitCompletion() {
        io_uring_cqe* cqe = nullptr;
        int responseCode = io_uring_wait_cqe(&ring_, &cqe);
        if (responseCode < 0) {
            throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::strerror(-responseCode));
        }

        Completion out;
        out.userData = cqe->user_data;
        out.responseCode = cqe->res;

        io_uring_cqe_seen(&ring_, cqe);
        return out;
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
    static bool isAligned(const void* p, std::size_t align) {
        auto v = reinterpret_cast<std::uintptr_t>(p);
        return (v % align) == 0;
    }

    void moveFrom(UringDirect&& other) noexcept {
        ring_ = other.ring_;
        ringInited_ = other.ringInited_;
        other.ringInited_ = false;

        fd_ = other.fd_;
        other.fd_ = -1;

        iovecs_ = std::move(other.iovecs_);

        buffersRegistered_ = other.buffersRegistered_;
        other.buffersRegistered_ = false;

        fileRegistered_ = other.fileRegistered_;
        other.fileRegistered_ = false;
    }

    uint64_t nextSeq() { return nextIssueSeq_++; }

    io_uring ring_{};
    bool ringInited_ = false;

    int fd_ = -1;

    std::vector<iovec> iovecs_;
    bool buffersRegistered_ = false;
    bool fileRegistered_ = false;

    std::unordered_map<uint64_t, Completion> pending_;
    static constexpr uint64_t SEQUENCE_START = 1000;
    uint64_t nextIssueSeq_ = SEQUENCE_START;
    uint64_t nextDeliverSeq_ = SEQUENCE_START;
};
