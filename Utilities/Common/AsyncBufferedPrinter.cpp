#include "Utilities/Common/AsyncBufferedPrinter.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace Thor {
namespace {

constexpr const char* kRed = "\x1b[31m";
constexpr const char* kBoldRed = "\x1b[1;31m";
constexpr const char* kReset = "\x1b[0m";
constexpr const char* kSaturationWarning =
    "WARNING trainer logging: async output queue filled; logging has been detected to be the performance bottleneck, "
    "so the training thread blocked while queued logs were dumped.\n";

}  // namespace

AsyncBufferedPrinter::AsyncBufferedPrinter() : AsyncBufferedPrinter(Options{}) {}

AsyncBufferedPrinter::AsyncBufferedPrinter(Options options) : options(std::move(options)) {
    if (this->options.flushInterval.count() < 0) {
        this->options.flushInterval = DEFAULT_FLUSH_INTERVAL;
    }
    if (this->options.flushBytes == 0) {
        this->options.flushBytes = DEFAULT_FLUSH_BYTES;
    }
    if (this->options.maxQueuedBytes < this->options.flushBytes) {
        this->options.maxQueuedBytes = this->options.flushBytes;
    }

    stdoutBuffer.reserve(this->options.maxQueuedBytes);
    stderrBuffer.reserve(this->options.maxQueuedBytes);
    worker = std::thread([this]() { workerLoop(); });
}

AsyncBufferedPrinter::~AsyncBufferedPrinter() { close(); }

void AsyncBufferedPrinter::write(std::string_view text, AsyncBufferedPrinterDestination destination) {
    if (text.empty()) {
        return;
    }

    const std::size_t bytesToQueue = queuedBytesFor(text, destination);
    std::unique_lock<std::mutex> lock(mutex);
    if (closeRequested) {
        return;
    }

    while (!closeRequested && queuedBytes > 0 && queuedBytes + bytesToQueue > options.maxQueuedBytes) {
        requestSaturationWarningLocked();
        workAvailable.notify_one();
        spaceOrFlushAvailable.wait(lock, [&]() { return closeRequested || queuedBytes == 0 || queuedBytes + bytesToQueue <= options.maxQueuedBytes; });
    }

    if (closeRequested) {
        return;
    }

    appendLocked(text, destination);
    if (queuedBytes >= options.flushBytes) {
        flushRequested = true;
    }
    workAvailable.notify_one();
}

void AsyncBufferedPrinter::writeLine(std::string_view text, AsyncBufferedPrinterDestination destination) {
    const std::size_t stdoutBytes = text.size() + 1;
    const std::size_t stderrBytes = (destination == AsyncBufferedPrinterDestination::STDOUT_AND_STDERR)
                                        ? std::strlen(kRed) + text.size() + 1 + std::strlen(kReset)
                                        : 0;
    const std::size_t bytesToQueue = stdoutBytes + stderrBytes;
    std::unique_lock<std::mutex> lock(mutex);
    if (closeRequested) {
        return;
    }

    while (!closeRequested && queuedBytes > 0 && queuedBytes + bytesToQueue > options.maxQueuedBytes) {
        requestSaturationWarningLocked();
        workAvailable.notify_one();
        spaceOrFlushAvailable.wait(lock, [&]() { return closeRequested || queuedBytes == 0 || queuedBytes + bytesToQueue <= options.maxQueuedBytes; });
    }

    if (closeRequested) {
        return;
    }

    stdoutBuffer.append(text.data(), text.size());
    stdoutBuffer.push_back('\n');
    queuedBytes += stdoutBytes;
    if (destination == AsyncBufferedPrinterDestination::STDOUT_AND_STDERR) {
        const std::size_t before = stderrBuffer.size();
        appendRed(stderrBuffer, text);
        stderrBuffer.push_back('\n');
        queuedBytes += stderrBuffer.size() - before;
    }

    if (queuedBytes >= options.flushBytes) {
        flushRequested = true;
    }
    workAvailable.notify_one();
}

void AsyncBufferedPrinter::flush() {
    std::unique_lock<std::mutex> lock(mutex);
    if (closeRequested) {
        return;
    }
    flushRequested = true;
    workAvailable.notify_one();
    spaceOrFlushAvailable.wait(lock, [&]() { return queuedBytes == 0 && !workerWriting; });
}

void AsyncBufferedPrinter::close() {
    std::thread workerToJoin;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closeRequested && !worker.joinable()) {
            return;
        }
        closeRequested = true;
        flushRequested = true;
    }
    workAvailable.notify_one();
    if (worker.joinable()) {
        workerToJoin = std::move(worker);
    }
    if (workerToJoin.joinable()) {
        workerToJoin.join();
    }
}

void AsyncBufferedPrinter::workerLoop() noexcept {
    std::string localStdout;
    std::string localStderr;
    localStdout.reserve(options.maxQueuedBytes);
    localStderr.reserve(options.maxQueuedBytes);

    while (true) {
        bool emitSaturationWarning = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!closeRequested && queuedBytes == 0 && !flushRequested && !saturationWarningRequested) {
                workAvailable.wait(lock, [&]() { return closeRequested || queuedBytes > 0 || flushRequested || saturationWarningRequested; });
            } else if (!closeRequested && queuedBytes > 0 && !flushRequested && !saturationWarningRequested && queuedBytes < options.flushBytes) {
                workAvailable.wait_for(lock, options.flushInterval, [&]() {
                    return closeRequested || flushRequested || saturationWarningRequested || queuedBytes >= options.flushBytes;
                });
            }

            if (queuedBytes == 0 && !saturationWarningRequested) {
                flushRequested = false;
                spaceOrFlushAvailable.notify_all();
                if (closeRequested) {
                    break;
                }
                continue;
            }

            localStdout.clear();
            localStderr.clear();
            localStdout.swap(stdoutBuffer);
            localStderr.swap(stderrBuffer);
            queuedBytes = 0;
            emitSaturationWarning = saturationWarningRequested && !saturationWarningEmitted;
            if (saturationWarningRequested) {
                saturationWarningEmitted = true;
                saturationWarningRequested = false;
            }
            flushRequested = false;
            workerWriting = true;
            spaceOrFlushAvailable.notify_all();
        }

        if (!localStdout.empty()) {
            writeAll(options.stdoutFd, localStdout.data(), localStdout.size());
        }
        if (!localStderr.empty()) {
            writeAll(options.stderrFd, localStderr.data(), localStderr.size());
        }
        if (emitSaturationWarning) {
            std::string warning;
            warning.reserve(std::strlen(kBoldRed) + std::strlen(kSaturationWarning) + std::strlen(kReset));
            warning.append(kBoldRed);
            warning.append(kSaturationWarning);
            warning.append(kReset);
            writeAll(options.stdoutFd, warning.data(), warning.size());
            writeAll(options.stderrFd, warning.data(), warning.size());
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            workerWriting = false;
        }
        spaceOrFlushAvailable.notify_all();
    }
}

void AsyncBufferedPrinter::appendLocked(std::string_view text, AsyncBufferedPrinterDestination destination) {
    stdoutBuffer.append(text.data(), text.size());
    queuedBytes += text.size();
    if (destination == AsyncBufferedPrinterDestination::STDOUT_AND_STDERR) {
        const std::size_t before = stderrBuffer.size();
        appendRed(stderrBuffer, text);
        queuedBytes += stderrBuffer.size() - before;
    }
}

void AsyncBufferedPrinter::requestSaturationWarningLocked() { saturationWarningRequested = true; }

std::size_t AsyncBufferedPrinter::queuedBytesFor(std::string_view text, AsyncBufferedPrinterDestination destination) const {
    std::size_t bytes = text.size();
    if (destination == AsyncBufferedPrinterDestination::STDOUT_AND_STDERR) {
        bytes += std::strlen(kRed) + text.size() + std::strlen(kReset);
    }
    return bytes;
}

void AsyncBufferedPrinter::writeAll(int fd, const char* data, std::size_t bytes) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    std::size_t writtenTotal = 0;
    while (writtenTotal < bytes) {
        const ssize_t written = ::write(fd, data + writtenTotal, bytes - writtenTotal);
        if (written > 0) {
            writtenTotal += static_cast<std::size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
#else
    (void)fd;
    (void)data;
    (void)bytes;
#endif
}

void AsyncBufferedPrinter::appendRed(std::string& out, std::string_view text) {
    out.append(kRed);

    for (std::size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch == 0x1b && i + 1 < text.size() && text[i + 1] == '[') {
            i += 2;
            while (i < text.size()) {
                const unsigned char ansiCh = static_cast<unsigned char>(text[i++]);
                if (ansiCh >= 0x40 && ansiCh <= 0x7e) {
                    break;
                }
            }
            continue;
        }
        out.push_back(text[i++]);
    }

    out.append(kReset);
}

}  // namespace Thor
