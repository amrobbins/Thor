#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
