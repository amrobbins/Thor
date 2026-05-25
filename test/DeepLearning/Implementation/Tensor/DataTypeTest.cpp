#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include "gtest/gtest.h"

#include <type_traits>

TEST(DataType, NamespaceAliasesUseTheImplementationEnum) {
    static_assert(std::is_same_v<Thor::DataType, ThorImplementation::DataType>);

    EXPECT_EQ(Thor::DataType::FP32, ThorImplementation::DataType::FP32);
    EXPECT_EQ(Thor::DataType::BF16, ThorImplementation::DataType::BF16);

    Thor::Tensor tensor(Thor::DataType::FP16, {2, 3});
    EXPECT_EQ(tensor.getDataType(), Thor::DataType::FP16);
}
