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
 * stores the expected value into mapped pinned host memory. Unlike a spinning
 * kernel, the wait does not occupy an SM, and release does not depend on any
 * CUDA stream being scheduled.
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
    uint32_t* released_h = nullptr;
    uint32_t* released_d = nullptr;
    std::optional<Stream> gatedStream;
    Event completionEvent;
    bool released = false;
};

}  // namespace ThorImplementation::Test
