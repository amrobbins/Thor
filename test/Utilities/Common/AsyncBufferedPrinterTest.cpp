#include "Utilities/Common/AsyncBufferedPrinter.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Thor {
namespace {

#if defined(__unix__) || defined(__APPLE__)

class Pipe {
   public:
    Pipe() {
        int fds[2];
        if (::pipe(fds) != 0) {
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
        }
        readFd_ = fds[0];
        writeFd_ = fds[1];
    }

    ~Pipe() {
        closeRead();
        closeWrite();
    }

    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    int readFd() const { return readFd_; }
    int writeFd() const { return writeFd_; }

    void closeRead() {
        if (readFd_ >= 0) {
            ::close(readFd_);
            readFd_ = -1;
        }
    }

    void closeWrite() {
        if (writeFd_ >= 0) {
            ::close(writeFd_);
            writeFd_ = -1;
        }
    }

    std::string readAllAfterWriterClosed() {
        closeWrite();
        std::string out;
        char buffer[4096];
        while (true) {
            const ssize_t n = ::read(readFd_, buffer, sizeof(buffer));
            if (n > 0) {
                out.append(buffer, static_cast<std::size_t>(n));
                continue;
            }
            if (n == 0) {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
        }
        return out;
    }

    std::string readAvailableNonblocking() const {
        const int oldFlags = ::fcntl(readFd_, F_GETFL, 0);
        if (oldFlags < 0) {
            throw std::runtime_error(std::string("fcntl(F_GETFL) failed: ") + std::strerror(errno));
        }
        if (::fcntl(readFd_, F_SETFL, oldFlags | O_NONBLOCK) != 0) {
            throw std::runtime_error(std::string("fcntl(F_SETFL) failed: ") + std::strerror(errno));
        }

        std::string out;
        char buffer[4096];
        while (true) {
            const ssize_t n = ::read(readFd_, buffer, sizeof(buffer));
            if (n > 0) {
                out.append(buffer, static_cast<std::size_t>(n));
                continue;
            }
            if (n == 0 || (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            const int savedErrno = errno;
            ::fcntl(readFd_, F_SETFL, oldFlags);
            throw std::runtime_error(std::string("read failed: ") + std::strerror(savedErrno));
        }

        if (::fcntl(readFd_, F_SETFL, oldFlags) != 0) {
            throw std::runtime_error(std::string("fcntl(restore) failed: ") + std::strerror(errno));
        }
        return out;
    }

   private:
    int readFd_ = -1;
    int writeFd_ = -1;
};

AsyncBufferedPrinter::Options pipeOptions(Pipe& stdoutPipe, Pipe& stderrPipe) {
    AsyncBufferedPrinter::Options options;
    options.flushInterval = std::chrono::hours(1);
    options.flushBytes = 64 * 1024;
    options.maxQueuedBytes = 1024 * 1024;
    options.stdoutFd = stdoutPipe.writeFd();
    options.stderrFd = stderrPipe.writeFd();
    return options;
}

TEST(AsyncBufferedPrinter, CloseFlushesStdoutOnlyLines) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter printer(pipeOptions(stdoutPipe, stderrPipe));
    printer.writeLine("alpha");
    printer.writeLine("beta");
    printer.close();

    EXPECT_EQ(stdoutPipe.readAllAfterWriterClosed(), "alpha\nbeta\n");
    EXPECT_EQ(stderrPipe.readAllAfterWriterClosed(), "");
}

TEST(AsyncBufferedPrinter, FlushPublishesPendingOutputWithoutClosingPrinter) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter printer(pipeOptions(stdoutPipe, stderrPipe));
    printer.write("partial");
    printer.flush();

    EXPECT_EQ(stdoutPipe.readAvailableNonblocking(), "partial");
    EXPECT_EQ(stderrPipe.readAvailableNonblocking(), "");

    printer.writeLine(" tail");
    printer.close();

    EXPECT_EQ(stdoutPipe.readAllAfterWriterClosed(), " tail\n");
    EXPECT_EQ(stderrPipe.readAllAfterWriterClosed(), "");
}

TEST(AsyncBufferedPrinter, StdoutAndStderrMirrorsWriteStdoutPlainAndStderrRed) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter printer(pipeOptions(stdoutPipe, stderrPipe));
    printer.writeLine("important", AsyncBufferedPrinterDestination::STDOUT_AND_STDERR);
    printer.close();

    EXPECT_EQ(stdoutPipe.readAllAfterWriterClosed(), "important\n");
    EXPECT_EQ(stderrPipe.readAllAfterWriterClosed(), "\x1b[31mimportant\x1b[0m\n");
}

TEST(AsyncBufferedPrinter, StdoutPreservesAnsiAndStderrMirrorStripsNestedAnsiBeforeRedWrapping) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter printer(pipeOptions(stdoutPipe, stderrPipe));
    printer.writeLine("pre \x1b[1mhot\x1b[0m post", AsyncBufferedPrinterDestination::STDOUT_AND_STDERR);
    printer.close();

    EXPECT_EQ(stdoutPipe.readAllAfterWriterClosed(), "pre \x1b[1mhot\x1b[0m post\n");
    EXPECT_EQ(stderrPipe.readAllAfterWriterClosed(), "\x1b[31mpre hot post\x1b[0m\n");
}

TEST(AsyncBufferedPrinter, FlushBytesThresholdWakesWriterBeforeClose) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter::Options options = pipeOptions(stdoutPipe, stderrPipe);
    options.flushInterval = std::chrono::hours(1);
    options.flushBytes = 8;
    options.maxQueuedBytes = 1024;

    AsyncBufferedPrinter printer(options);
    printer.write("1234");
    EXPECT_EQ(stdoutPipe.readAvailableNonblocking(), "");

    printer.write("5678");
    printer.flush();
    EXPECT_EQ(stdoutPipe.readAvailableNonblocking(), "12345678");

    printer.close();
    EXPECT_EQ(stdoutPipe.readAllAfterWriterClosed(), "");
    EXPECT_EQ(stderrPipe.readAllAfterWriterClosed(), "");
}

TEST(AsyncBufferedPrinter, QueueSaturationBlocksLosslesslyAndEmitsWarningToBothStreams) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter::Options options = pipeOptions(stdoutPipe, stderrPipe);
    options.flushInterval = std::chrono::hours(1);
    options.flushBytes = 64;
    options.maxQueuedBytes = 64;

    AsyncBufferedPrinter printer(options);
    const std::string first(40, 'a');
    const std::string second(40, 'b');

    printer.write(first);
    printer.write(second);
    printer.close();

    const std::string stdoutText = stdoutPipe.readAllAfterWriterClosed();
    const std::string stderrText = stderrPipe.readAllAfterWriterClosed();

    EXPECT_NE(stdoutText.find(first), std::string::npos);
    EXPECT_NE(stdoutText.find(second), std::string::npos);
    EXPECT_NE(stdoutText.find("\x1b[1;31mWARNING trainer logging: async output queue filled;"), std::string::npos);
    EXPECT_NE(stdoutText.find("\x1b[0m"), std::string::npos);

    EXPECT_NE(stderrText.find("\x1b[1;31mWARNING trainer logging: async output queue filled;"), std::string::npos);
    EXPECT_NE(stderrText.find("\x1b[0m"), std::string::npos);
}

TEST(AsyncBufferedPrinter, CloseIsIdempotentAndWritesAfterCloseAreIgnored) {
    Pipe stdoutPipe;
    Pipe stderrPipe;

    AsyncBufferedPrinter printer(pipeOptions(stdoutPipe, stderrPipe));
    printer.writeLine("before");
    printer.close();
    printer.close();
    printer.writeLine("after");

    EXPECT_EQ(stdoutPipe.readAllAfterWriterClosed(), "before\n");
    EXPECT_EQ(stderrPipe.readAllAfterWriterClosed(), "");
}

#else

TEST(AsyncBufferedPrinter, UnixPipeTestsRequireUnixFileDescriptors) { GTEST_SKIP() << "AsyncBufferedPrinter fd capture tests require Unix file descriptors."; }

#endif

}  // namespace
}  // namespace Thor
