#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace Thor {

enum class AsyncBufferedPrinterDestination { STDOUT, STDOUT_AND_STDERR };

class AsyncBufferedPrinter {
   public:
    static constexpr std::chrono::milliseconds DEFAULT_FLUSH_INTERVAL{25};
    static constexpr std::size_t DEFAULT_FLUSH_BYTES = 64 * 1024;
    static constexpr std::size_t DEFAULT_MAX_QUEUED_BYTES = 1024 * 1024;

    struct Options {
        // Wake the writer thread at least this often while output is queued.
        std::chrono::milliseconds flushInterval{DEFAULT_FLUSH_INTERVAL};

        // Wake the writer thread immediately once pending output reaches this size.
        std::size_t flushBytes = DEFAULT_FLUSH_BYTES;

        // Lossless backpressure threshold: producer threads block once queued output reaches this size.
        std::size_t maxQueuedBytes = DEFAULT_MAX_QUEUED_BYTES;

        // Output file descriptors. Production uses stdout/stderr; tests can pass pipes.
        int stdoutFd = 1;
        int stderrFd = 2;
    };

    AsyncBufferedPrinter();
    explicit AsyncBufferedPrinter(Options options);
    ~AsyncBufferedPrinter();

    AsyncBufferedPrinter(const AsyncBufferedPrinter&) = delete;
    AsyncBufferedPrinter& operator=(const AsyncBufferedPrinter&) = delete;

    void write(std::string_view text, AsyncBufferedPrinterDestination destination = AsyncBufferedPrinterDestination::STDOUT);
    void writeLine(std::string_view text, AsyncBufferedPrinterDestination destination = AsyncBufferedPrinterDestination::STDOUT);

    void flush();
    void close();

   private:
    void workerLoop() noexcept;
    void appendLocked(std::string_view text, AsyncBufferedPrinterDestination destination);
    void requestSaturationWarningLocked();
    [[nodiscard]] std::size_t queuedBytesFor(std::string_view text, AsyncBufferedPrinterDestination destination) const;
    static void writeAll(int fd, const char* data, std::size_t bytes) noexcept;
    static void appendRed(std::string& out, std::string_view text);

    Options options;
    std::mutex mutex;
    std::condition_variable workAvailable;
    std::condition_variable spaceOrFlushAvailable;
    std::thread worker;

    std::string stdoutBuffer;
    std::string stderrBuffer;
    std::size_t queuedBytes = 0;

    bool flushRequested = false;
    bool closeRequested = false;
    bool workerWriting = false;
    bool saturationWarningRequested = false;
    bool saturationWarningEmitted = false;
};

}  // namespace Thor
