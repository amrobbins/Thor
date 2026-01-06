#include "DeepLearning/Api/Initializers/Glorot.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Initializers, GlorotBuilds) {
    srand(time(nullptr));

    ThorImplementation::Glorot::Mode mode =
        rand() % 2 == 0 ? ThorImplementation::Glorot::Mode::UNIFORM : ThorImplementation::Glorot::Mode::NORMAL;

    shared_ptr<Initializer> initializer = Glorot::Builder().mode(mode).build();

    ASSERT_TRUE(initializer->isInitialized());
    ASSERT_TRUE(initializer->clone()->isInitialized());

    shared_ptr<Glorot> glorot = dynamic_pointer_cast<Glorot>(initializer);
    ASSERT_TRUE(glorot->isInitialized());
    ASSERT_EQ(glorot->getMode(), mode);
    ASSERT_TRUE(glorot->clone()->isInitialized());
    ASSERT_EQ(dynamic_pointer_cast<Glorot>(glorot->clone())->getMode(), mode);
}

TEST(Initializers, GlorotSerializeDeserialize) {
    srand(time(nullptr));

    ThorImplementation::Glorot::Mode mode =
        rand() % 2 == 0 ? ThorImplementation::Glorot::Mode::UNIFORM : ThorImplementation::Glorot::Mode::NORMAL;

    shared_ptr<Initializer> glorot = Glorot::Builder().mode(mode).build();

    Stream stream(0);

    json glorotJ = glorot->serialize();

    // printf("%s\n", glorotJ.dump(4).c_str());

    ASSERT_EQ(glorotJ.at("initializer_type").get<string>(), "glorot");
    ASSERT_EQ(glorotJ.at("version").get<string>(), glorot->getVersion());
    ASSERT_EQ(glorotJ.at("mode").get<ThorImplementation::Glorot::Mode>(), mode);

    shared_ptr<Initializer> initializerDeserialized = Initializer::deserialize(glorotJ);
    ASSERT_TRUE(initializerDeserialized->isInitialized());

    shared_ptr<Glorot> glorotDeserialized = dynamic_pointer_cast<Glorot>(initializerDeserialized);
    ASSERT_TRUE(glorotDeserialized->isInitialized());
    ASSERT_EQ(glorotDeserialized->getVersion(), glorot->getVersion());
    ASSERT_EQ(glorotDeserialized->getMode(), mode);
}
