#include "DeepLearning/Api/Data/DatasetAccessPolicy.h"
#include "DeepLearning/Api/Training/WindowedDeviceCache.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(WindowedDeviceCacheTest, NamesRoundTripAndRejectUnknownValues) {
    EXPECT_STREQ(Thor::windowedDeviceCacheName(Thor::WindowedDeviceCache::OFF), "off");
    EXPECT_STREQ(Thor::windowedDeviceCacheName(Thor::WindowedDeviceCache::AUTO), "auto");
    EXPECT_STREQ(Thor::windowedDeviceCacheName(Thor::WindowedDeviceCache::REQUIRED), "required");

    EXPECT_EQ(Thor::windowedDeviceCacheFromName("off"), Thor::WindowedDeviceCache::OFF);
    EXPECT_EQ(Thor::windowedDeviceCacheFromName("auto"), Thor::WindowedDeviceCache::AUTO);
    EXPECT_EQ(Thor::windowedDeviceCacheFromName("required"), Thor::WindowedDeviceCache::REQUIRED);
    EXPECT_THROW((void)Thor::windowedDeviceCacheFromName("sometimes"), std::runtime_error);
}

TEST(WindowedDeviceCacheTest, DatasetAccessPolicyDefaultsToAuto) {
    const Thor::DatasetAccessPolicy policy;
    EXPECT_EQ(policy.windowedDeviceCache, Thor::WindowedDeviceCache::AUTO);
}
