#pragma once

#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <optional>

namespace ThorImplementation::Test {

/**
 * Test-only GPU-side stream gate.
 *
 * The gated stream waits on a CUDA stream memory operation until release()
 * writes the expected value from an independent control stream. Unlike a
 * spinning kernel, the wait does not occupy an SM and therefore cannot prevent
 * the release operation from being scheduled.
 */
class DeviceStreamGate {
   public:
    explicit DeviceStreamGate(int32_t gpuNum);
    ~DeviceStreamGate();

    DeviceStreamGate(const DeviceStreamGate&) = delete;
    DeviceStreamGate& operator=(const DeviceStreamGate&) = delete;
    DeviceStreamGate(DeviceStreamGate&&) = delete;
    DeviceStreamGate& operator=(DeviceStreamGate&&) = delete;

    void enqueue(const Stream& stream);
    void release();
    [[nodiscard]] bool isComplete();

   private:
    void releaseNoThrow() noexcept;

    int32_t gpuNum;
    uint32_t* released_d = nullptr;
    Stream controlStream;
    std::optional<Stream> gatedStream;
    Event completionEvent;
    bool released = false;
};

}  // namespace ThorImplementation::Test
