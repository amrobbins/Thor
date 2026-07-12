#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TestStampedNetwork final : public ThorImplementation::StampedNetwork {
   public:
    void addOtherLayer(const std::shared_ptr<ThorImplementation::Layer>& layer) {
        otherLayersShared.push_back(layer);
        otherLayers.push_back(layer.get());
    }

    void clearForTesting() { clear(); }

    [[nodiscard]] bool hasOtherLayers() const {
        return !otherLayers.empty() || !otherLayersShared.empty();
    }
};

class CleanupProbeLayer final : public ThorImplementation::Layer {
   public:
    CleanupProbeLayer(bool& cleaned, bool throwDuringCleanup)
        : cleaned(cleaned), throwDuringCleanup(throwDuringCleanup) {}

    void cleanup() override {
        cleaned = true;
        ThorImplementation::Layer::cleanup();
        if (throwDuringCleanup) {
            throw std::runtime_error("intentional cleanup failure");
        }
    }

   protected:
    void infer(std::optional<ThorImplementation::Tensor>,
               std::optional<ThorImplementation::Tensor>,
               Stream) override {}

    void backProp(std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  Stream) override {}

   private:
    bool& cleaned;
    bool throwDuringCleanup;
};

TEST(StampedNetworkCleanupTest, ClearContinuesAfterOneLayerCleanupThrows) {
    bool firstCleaned = false;
    bool secondCleaned = false;
    auto first = std::make_shared<CleanupProbeLayer>(firstCleaned, true);
    auto second = std::make_shared<CleanupProbeLayer>(secondCleaned, false);

    TestStampedNetwork stamped;
    stamped.addOtherLayer(first);
    stamped.addOtherLayer(second);

    EXPECT_THROW(stamped.clearForTesting(), std::runtime_error);
    EXPECT_TRUE(firstCleaned);
    EXPECT_TRUE(secondCleaned);
    EXPECT_FALSE(stamped.hasOtherLayers());
}

TEST(StampedNetworkCleanupTest, PlacedNetworkDestructorContinuesAndSuppressesCleanupErrors) {
    bool firstCleaned = false;
    bool secondCleaned = false;
    auto first = std::make_shared<CleanupProbeLayer>(firstCleaned, true);
    auto second = std::make_shared<CleanupProbeLayer>(secondCleaned, false);

    TestStampedNetwork stamped;
    stamped.addOtherLayer(first);
    stamped.addOtherLayer(second);
    std::vector<ThorImplementation::StampedNetwork> stamps{stamped};
    Thor::Network network("cleanup_test_network");

    EXPECT_NO_THROW({
        Thor::PlacedNetwork placed("cleanup_test_network", network, stamps);
    });
    EXPECT_TRUE(firstCleaned);
    EXPECT_TRUE(secondCleaned);
}

}  // namespace
