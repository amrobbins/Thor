#include "Utilities/Common/PersistingL2Cache.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

using namespace ThorImplementation;

namespace {

bool gpuAvailable() { return MachineEvaluator::instance().getNumGpus() > 0; }

class ScopedDeviceAllocation {
   public:
    explicit ScopedDeviceAllocation(size_t bytes) { EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess); }
    ~ScopedDeviceAllocation() {
        if (ptr != nullptr)
            EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }

    ScopedDeviceAllocation(const ScopedDeviceAllocation&) = delete;
    ScopedDeviceAllocation& operator=(const ScopedDeviceAllocation&) = delete;

    void* get() const { return ptr; }

   private:
    void* ptr = nullptr;
};

}  // namespace

TEST(PersistingL2Cache, NegativeGpuNumberIsReportedWithoutThrowing) {
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(-1);
    EXPECT_FALSE(capabilities.query_succeeded);
    EXPECT_FALSE(capabilities.supported);
    EXPECT_EQ(capabilities.cuda_status, cudaErrorInvalidValue);
    EXPECT_NE(capabilities.detail.find("gpu_num must be non-negative"), std::string::npos);

    const PersistingL2OperationResult result = trySetPersistingL2SetAsideBytes(-1, 0);
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);
    EXPECT_FALSE(result.succeeded());
}

TEST(PersistingL2Cache, StreamRejectsInvalidWindowArgumentsBeforeCudaConfiguration) {
    Stream uninitialized;
    const int value = 7;
    PersistingL2OperationResult result = uninitialized.trySetPersistingL2AccessPolicyWindow(&value, sizeof(value), 1.0f);
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);

    if (!gpuAvailable())
        GTEST_SKIP() << "remaining validation requires an initialized CUDA stream";

    ScopedGpu scopedGpu(0);
    Stream stream(0);
    result = stream.trySetPersistingL2AccessPolicyWindow(nullptr, sizeof(value), 1.0f);
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);

    result = stream.trySetPersistingL2AccessPolicyWindow(&value, 0, 1.0f);
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);

    result = stream.trySetPersistingL2AccessPolicyWindow(&value, sizeof(value), -0.01f);
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);

    result = stream.trySetPersistingL2AccessPolicyWindow(&value, sizeof(value), 1.01f);
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);

    result = stream.trySetPersistingL2AccessPolicyWindow(
        &value, sizeof(value), std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(result.status, PersistingL2OperationStatus::INVALID_ARGUMENT);
}

TEST(PersistingL2Cache, CapabilityQueryMatchesCudaDeviceProperties) {
    if (!gpuAvailable())
        GTEST_SKIP() << "persisting-L2 capability test requires a GPU";

    ScopedGpu scopedGpu(0);
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);

    const bool expectedHardwareSupport = properties.major >= 8 && properties.persistingL2CacheMaxSize > 0 &&
                                         properties.accessPolicyMaxWindowSize > 0;
    size_t currentSetAside = 0;
    const cudaError_t getLimitStatus =
        expectedHardwareSupport ? cudaDeviceGetLimit(&currentSetAside, cudaLimitPersistingL2CacheSize) : cudaSuccess;

    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    EXPECT_EQ(capabilities.compute_capability_major, properties.major);
    EXPECT_EQ(capabilities.compute_capability_minor, properties.minor);
    EXPECT_EQ(capabilities.l2_bytes, static_cast<uint64_t>(std::max(properties.l2CacheSize, 0)));
    EXPECT_EQ(capabilities.max_persisting_bytes,
              properties.persistingL2CacheMaxSize > 0 ? static_cast<uint64_t>(properties.persistingL2CacheMaxSize) : 0u);
    EXPECT_EQ(capabilities.max_access_policy_window_bytes,
              properties.accessPolicyMaxWindowSize > 0 ? static_cast<uint64_t>(properties.accessPolicyMaxWindowSize) : 0u);

    if (!expectedHardwareSupport) {
        EXPECT_TRUE(capabilities.query_succeeded);
        EXPECT_FALSE(capabilities.supported);
        EXPECT_FALSE(capabilities.detail.empty());
        return;
    }
    if (getLimitStatus != cudaSuccess) {
        EXPECT_FALSE(capabilities.query_succeeded);
        EXPECT_FALSE(capabilities.supported);
        EXPECT_EQ(capabilities.cuda_status, getLimitStatus);
        return;
    }

    ASSERT_TRUE(capabilities.query_succeeded) << capabilities.detail;
    EXPECT_TRUE(capabilities.supported) << capabilities.detail;
    EXPECT_EQ(capabilities.current_persisting_bytes, static_cast<uint64_t>(currentSetAside));
}

TEST(PersistingL2Cache, OversizedSetAsideAndAccessWindowAreRejectedBeforeMutation) {
    if (!gpuAvailable())
        GTEST_SKIP() << "persisting-L2 bounds test requires a GPU";

    ScopedGpu scopedGpu(0);
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    if (capabilities.max_persisting_bytes < std::numeric_limits<uint64_t>::max()) {
        const PersistingL2OperationResult setAside =
            trySetPersistingL2SetAsideBytes(0, capabilities.max_persisting_bytes + 1);
        EXPECT_EQ(setAside.status, PersistingL2OperationStatus::INVALID_ARGUMENT);
    }

    Stream stream(0);
    const int value = 7;
    if (capabilities.max_access_policy_window_bytes < std::numeric_limits<uint64_t>::max()) {
        const PersistingL2OperationResult window = stream.trySetPersistingL2AccessPolicyWindow(
            &value, capabilities.max_access_policy_window_bytes + 1, 1.0f);
        EXPECT_EQ(window.status, PersistingL2OperationStatus::INVALID_ARGUMENT);
    }
}

TEST(PersistingL2Cache, StreamCanApplyAndClearAccessPolicyWindowWhenSupported) {
    if (!gpuAvailable())
        GTEST_SKIP() << "persisting-L2 stream test requires a GPU";

    ScopedGpu scopedGpu(0);
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    constexpr size_t allocationBytes = 4096;
    if (capabilities.max_access_policy_window_bytes < allocationBytes)
        GTEST_SKIP() << "device access-policy window is smaller than test allocation";

    ScopedDeviceAllocation allocation(allocationBytes);
    ASSERT_NE(allocation.get(), nullptr);
    Stream stream(0);

    const PersistingL2OperationResult applied =
        stream.trySetPersistingL2AccessPolicyWindow(allocation.get(), allocationBytes, 0.75f);
    ASSERT_TRUE(applied.succeeded()) << applied.detail;

    cudaStreamAttrValue attribute{};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.base_ptr, allocation.get());
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, allocationBytes);
    EXPECT_FLOAT_EQ(attribute.accessPolicyWindow.hitRatio, 0.75f);
    EXPECT_EQ(attribute.accessPolicyWindow.hitProp, cudaAccessPropertyPersisting);
    EXPECT_EQ(attribute.accessPolicyWindow.missProp, cudaAccessPropertyStreaming);

    const PersistingL2OperationResult cleared = stream.tryClearPersistingL2AccessPolicyWindow();
    ASSERT_TRUE(cleared.succeeded()) << cleared.detail;

    attribute = {};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, 0u);
}

TEST(PersistingL2Cache, StreamCachesAccessPolicyWindowUntilCacheKeyChanges) {
    if (!gpuAvailable())
        GTEST_SKIP() << "persisting-L2 stream cache test requires a GPU";

    ScopedGpu scopedGpu(0);
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    constexpr size_t allocationBytes = 4096;
    if (capabilities.max_access_policy_window_bytes < allocationBytes)
        GTEST_SKIP() << "device access-policy window is smaller than test allocation";

    ScopedDeviceAllocation firstAllocation(allocationBytes);
    ScopedDeviceAllocation secondAllocation(allocationBytes);
    ASSERT_NE(firstAllocation.get(), nullptr);
    ASSERT_NE(secondAllocation.get(), nullptr);

    Stream stream(0);
    Stream sharedHandle = stream;

    constexpr uint64_t sourceIdentity = 41;
    constexpr uint64_t generation = 7;
    ASSERT_TRUE(stream.trySetPersistingL2AccessPolicyWindow(
        firstAllocation.get(), allocationBytes, 0.75f, sourceIdentity, generation).succeeded());

    // Out-of-band CUDA mutation is used only as a test probe: if the repeated
    // Stream request really is cached, it will not restore the raw attribute.
    ASSERT_TRUE(tryClearPersistingL2AccessPolicyWindow(0, stream.getStream()).succeeded());
    ASSERT_TRUE(sharedHandle.trySetPersistingL2AccessPolicyWindow(
        firstAllocation.get(), allocationBytes, 0.75f, sourceIdentity, generation).succeeded());

    cudaStreamAttrValue attribute{};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, 0u);

    // A manager rebalance generation change must refresh the stream even when
    // the physical region and ratio happen to be unchanged.
    ASSERT_TRUE(stream.trySetPersistingL2AccessPolicyWindow(
        firstAllocation.get(), allocationBytes, 0.75f, sourceIdentity, generation + 1).succeeded());
    attribute = {};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.base_ptr, firstAllocation.get());
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, allocationBytes);
    EXPECT_FLOAT_EQ(attribute.accessPolicyWindow.hitRatio, 0.75f);

    ASSERT_TRUE(tryClearPersistingL2AccessPolicyWindow(0, stream.getStream()).succeeded());
    ASSERT_TRUE(stream.trySetPersistingL2AccessPolicyWindow(
        firstAllocation.get(), allocationBytes, 0.75f, sourceIdentity + 1, generation + 1).succeeded());
    attribute = {};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.base_ptr, firstAllocation.get());
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, allocationBytes);

    ASSERT_TRUE(tryClearPersistingL2AccessPolicyWindow(0, stream.getStream()).succeeded());
    ASSERT_TRUE(stream.trySetPersistingL2AccessPolicyWindow(
        firstAllocation.get(), allocationBytes, 0.5f, sourceIdentity + 1, generation + 1).succeeded());
    attribute = {};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_FLOAT_EQ(attribute.accessPolicyWindow.hitRatio, 0.5f);

    ASSERT_TRUE(tryClearPersistingL2AccessPolicyWindow(0, stream.getStream()).succeeded());
    ASSERT_TRUE(stream.trySetPersistingL2AccessPolicyWindow(
        secondAllocation.get(), allocationBytes, 0.5f, sourceIdentity + 1, generation + 1).succeeded());
    attribute = {};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.base_ptr, secondAllocation.get());
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, allocationBytes);

    ASSERT_TRUE(stream.tryClearPersistingL2AccessPolicyWindow().succeeded());
}

TEST(PersistingL2Cache, StreamCachesSuccessfulClearUntilAnotherSetSucceeds) {
    if (!gpuAvailable())
        GTEST_SKIP() << "persisting-L2 stream clear-cache test requires a GPU";

    ScopedGpu scopedGpu(0);
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(0);
    if (!capabilities.query_succeeded || !capabilities.supported)
        GTEST_SKIP() << "persisting L2 unavailable: " << capabilities.detail;

    constexpr size_t allocationBytes = 4096;
    if (capabilities.max_access_policy_window_bytes < allocationBytes)
        GTEST_SKIP() << "device access-policy window is smaller than test allocation";

    ScopedDeviceAllocation allocation(allocationBytes);
    ASSERT_NE(allocation.get(), nullptr);
    Stream stream(0);

    ASSERT_TRUE(stream.trySetPersistingL2AccessPolicyWindow(
        allocation.get(), allocationBytes, 1.0f, 5, 11).succeeded());
    ASSERT_TRUE(stream.tryClearPersistingL2AccessPolicyWindow().succeeded());

    // Probe the cached "clear" state by changing the raw CUDA attribute behind
    // Stream's back. A second Stream clear should be a no-op.
    ASSERT_TRUE(trySetPersistingL2AccessPolicyWindow(
        0, stream.getStream(), allocation.get(), allocationBytes, 1.0f).succeeded());
    ASSERT_TRUE(stream.tryClearPersistingL2AccessPolicyWindow().succeeded());

    cudaStreamAttrValue attribute{};
    ASSERT_EQ(cudaStreamGetAttribute(stream.getStream(), cudaStreamAttributeAccessPolicyWindow, &attribute), cudaSuccess);
    EXPECT_EQ(attribute.accessPolicyWindow.base_ptr, allocation.get());
    EXPECT_EQ(attribute.accessPolicyWindow.num_bytes, allocationBytes);

    // Restore the real CUDA state because the out-of-band mutation above was
    // intentionally invisible to Stream's cache.
    ASSERT_TRUE(tryClearPersistingL2AccessPolicyWindow(0, stream.getStream()).succeeded());
}
