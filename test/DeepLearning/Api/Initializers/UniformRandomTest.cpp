#include "DeepLearning/Api/Initializers/UniformRandom.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Initializers, UniformRandomBuilds) {
    srand(time(nullptr));

    double minValue = 1.0 / double(1 + (rand() % 1000));
    double maxValue = 3 + (1.0 / double(1 + (rand() % 1000)));

    shared_ptr<Initializer> initializer = UniformRandom::Builder().minValue(minValue).maxValue(maxValue).build();

    ASSERT_TRUE(initializer->isInitialized());
    ASSERT_TRUE(initializer->clone()->isInitialized());

    shared_ptr<UniformRandom> uniformRandom = dynamic_pointer_cast<UniformRandom>(initializer);
    ASSERT_TRUE(uniformRandom->isInitialized());
    ASSERT_EQ(uniformRandom->getMinValue(), minValue);
    ASSERT_EQ(uniformRandom->getMaxValue(), maxValue);
    ASSERT_TRUE(uniformRandom->clone()->isInitialized());
    shared_ptr<UniformRandom> uniformRandomClone = dynamic_pointer_cast<UniformRandom>(uniformRandom->clone());
    ASSERT_TRUE(uniformRandomClone->clone()->isInitialized());
    ASSERT_EQ(uniformRandomClone->getMinValue(), minValue);
    ASSERT_EQ(uniformRandomClone->getMaxValue(), maxValue);
}

TEST(Initializers, UniformRandomSerializeDeserialize) {
    srand(time(nullptr));

    double minValue = 1.0 / double(1 + (rand() % 1000));
    double maxValue = 3 + (1.0 / double(1 + (rand() % 1000)));

    shared_ptr<Initializer> uniformRandom = UniformRandom::Builder().minValue(minValue).maxValue(maxValue).build();

    Stream stream(0);

    json uniformRandomJ = uniformRandom->serialize();

    // printf("%s\n", uniformRandomJ.dump(4).c_str());

    ASSERT_EQ(uniformRandomJ.at("initializer_type").get<string>(), "uniform_random");
    ASSERT_EQ(uniformRandomJ.at("version").get<string>(), uniformRandom->getVersion());
    ASSERT_EQ(uniformRandomJ.at("min_value").get<double>(), minValue);
    ASSERT_EQ(uniformRandomJ.at("max_value").get<double>(), maxValue);

    shared_ptr<Initializer> initializerDeserialized = Initializer::deserialize(uniformRandomJ);
    ASSERT_TRUE(initializerDeserialized->isInitialized());

    shared_ptr<UniformRandom> uniformRandomDeserialized = dynamic_pointer_cast<UniformRandom>(initializerDeserialized);
    ASSERT_TRUE(uniformRandomDeserialized->isInitialized());
    ASSERT_EQ(uniformRandomDeserialized->getVersion(), uniformRandom->getVersion());
    ASSERT_EQ(uniformRandomDeserialized->getMinValue(), minValue);
    ASSERT_EQ(uniformRandomDeserialized->getMaxValue(), maxValue);
}
