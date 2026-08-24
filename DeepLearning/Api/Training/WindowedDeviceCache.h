#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Thor {

/**
 * Policy for CUDA persisting-L2 treatment of compact, device-resident window
 * sources.
 *
 * This policy is intentionally narrow: it never marks ordinary device tensors,
 * direct fields, ordinary field-owned ragged storage, masks, or temporary
 * materialized batches as persisting. AUTO only considers the compact immutable
 * source payload behind dense or source-backed ragged windows.
 */
enum class WindowedDeviceCache {
    /** Do not request persisting-L2 treatment for compact window sources. */
    OFF,
    /** Best effort; unsupported/oversized sources continue with normal caching. */
    AUTO,
    /** Fail session creation if an eligible compact window source cannot be cached. */
    REQUIRED,
};

[[nodiscard]] const char *windowedDeviceCacheName(WindowedDeviceCache policy);
[[nodiscard]] WindowedDeviceCache windowedDeviceCacheFromName(std::string_view name);

/**
 * Per-session/report snapshot of the window-source persisting-L2 policy.
 *
 * eligibleSourceBytes counts unique compact source allocations referenced by
 * the session. activeUniqueBytes is device-wide manager telemetry and may
 * include sources leased by other concurrent sessions/runs on the same GPU.
 */
struct WindowedDeviceCacheReport {
    WindowedDeviceCache requested = WindowedDeviceCache::AUTO;
    bool attempted = false;
    bool used = false;
    std::string reason{};
    uint64_t eligibleSources = 0;
    uint64_t activeSources = 0;
    uint64_t eligibleSourceBytes = 0;
    uint64_t budgetBytes = 0;
    uint64_t maxAccessPolicyWindowBytes = 0;
    uint64_t activeUniqueBytes = 0;
    float hitRatio = 0.0f;
};

}  // namespace Thor
