// 1174: StampedCubReduction / StampedCubArgReduction::runOnTensorsForBenchmark, the hook the cold-L2 audits use to run
// one production stamp on rotating input and output views. A stamp run on a replacement input and output must write
// exactly what a stamp bound to those tensors writes, for direct and multi-pass plans, VALUE and ARG, FP32 and FP8, and
// must leave its own bound output untouched.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

struct Case {
    std::vector<uint64_t> dims;
    std::vector<uint32_t> axes;
    DataType dtype;
};

std::vector<Case> cases() {
    std::vector<Case> out{
        {{100003}, {0}, DataType::FP32},          // R
        {{257, 1031}, {1}, DataType::FP32},       // KR
        {{9, 4099, 37}, {1}, DataType::FP32},     // RK
        {{6, 5, 1000}, {1, 2}, DataType::FP32},   // two reduced axes
    };
#if THOR_CUB_ENABLE_FP8_TYPES
    out.push_back({{16, 65536, 3}, {1}, DataType::FP8_E4M3});  // deep R, small K: staged FP8 RK plus an FP32 pass
    out.push_back({{257, 1031}, {1}, DataType::FP8_E4M3});     // FP8 KR
    out.push_back({{6, 5, 1000}, {0, 2}, DataType::FP8_E5M2});  // multi-pass FP8
#endif
    return out;
}

std::vector<float> values(uint64_t n, std::mt19937& rng) {
    // Small integers: exact in FP8 and FP32, and sums stay exact, so results compare bit for bit.
    std::uniform_int_distribution<int> d(-3, 3);
    std::vector<float> v(n);
    for (float& x : v) x = static_cast<float>(d(rng));
    return v;
}

uint64_t product(const std::vector<uint64_t>& dims) {
    uint64_t p = 1;
    for (uint64_t d : dims) p *= d;
    return p;
}

}  // namespace

TEST(CubReductionBenchmarkRebind, ValueStampOnReplacementTensorsMatchesABoundStamp) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937 rng(1174);
    for (const Case& c : cases()) {
        SCOPED_TRACE(::testing::Message() << "dtype " << static_cast<int>(c.dtype) << " elements " << product(c.dims));
        Tensor bound_input = makeGpuTensor(values(product(c.dims), rng), c.dims, stream, c.dtype);
        Tensor other_input = makeGpuTensor(values(product(c.dims), rng), c.dims, stream, c.dtype);
        for (const CubReductionOp op : {CubReductionOp::Sum, CubReductionOp::Max}) {
            auto stamp = CubReduction(op, c.axes, DataType::FP32).stamp(bound_input, stream);
            auto reference = CubReduction(op, c.axes, DataType::FP32).stamp(other_input, stream);
            stamp->run();
            stream.synchronize();
            const std::vector<float> bound_before = copyGpuTensorAsFloat(stamp->getOutputTensor(), stream);

            Tensor replacement_output = stamp->getOutputTensor().clone();
            stamp->runOnTensorsForBenchmark(other_input, replacement_output, stream);
            reference->run();
            stream.synchronize();
            EXPECT_EQ(copyGpuTensorAsFloat(replacement_output, stream),
                      copyGpuTensorAsFloat(reference->getOutputTensor(), stream));
            EXPECT_EQ(copyGpuTensorAsFloat(stamp->getOutputTensor(), stream), bound_before);
        }
    }
}

TEST(CubReductionBenchmarkRebind, ArgStampOnReplacementTensorsMatchesABoundStamp) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937 rng(11742);
    for (const Case& c : cases()) {
        SCOPED_TRACE(::testing::Message() << "dtype " << static_cast<int>(c.dtype) << " elements " << product(c.dims));
        Tensor bound_input = makeGpuTensor(values(product(c.dims), rng), c.dims, stream, c.dtype);
        Tensor other_input = makeGpuTensor(values(product(c.dims), rng), c.dims, stream, c.dtype);
        for (const bool with_value : {false, true}) {
            CubArgReductionOutputOptions outputs;
            outputs.produce_value = with_value;
            if (with_value) outputs.value_output_dtype = DataType::FP32;
            outputs.produce_index = true;
            outputs.index_output_dtype = DataType::UINT32;
            auto stamp = CubArgReduction(CubArgReductionOp::ArgMax, c.axes, outputs).stamp(bound_input, stream);
            auto reference = CubArgReduction(CubArgReductionOp::ArgMax, c.axes, outputs).stamp(other_input, stream);
            stamp->run();
            stream.synchronize();
            const std::vector<uint64_t> bound_before = copyGpuTensorAsUnsigned(*stamp->getIndexOutputTensor(), stream);

            std::optional<Tensor> value = with_value ? std::optional<Tensor>(stamp->getValueOutputTensor()->clone())
                                                     : std::nullopt;
            std::optional<Tensor> index = stamp->getIndexOutputTensor()->clone();
            stamp->runOnTensorsForBenchmark(other_input, value, index, stream);
            reference->run();
            stream.synchronize();
            EXPECT_EQ(copyGpuTensorAsUnsigned(*index, stream),
                      copyGpuTensorAsUnsigned(*reference->getIndexOutputTensor(), stream));
            if (with_value) {
                EXPECT_EQ(copyGpuTensorAsFloat(*value, stream),
                          copyGpuTensorAsFloat(*reference->getValueOutputTensor(), stream));
            }
            EXPECT_EQ(copyGpuTensorAsUnsigned(*stamp->getIndexOutputTensor(), stream), bound_before);
        }
    }
}

TEST(CubReductionBenchmarkRebind, MismatchedReplacementsAreRejected) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::mt19937 rng(11743);
    Tensor input = makeGpuTensor(values(257 * 1031, rng), {257, 1031}, stream, DataType::FP32);
    Tensor smaller = makeGpuTensor(values(257 * 1030, rng), {257, 1030}, stream, DataType::FP32);
    auto stamp = CubReduction(CubReductionOp::Sum, 1U, DataType::FP32).stamp(input, stream);
    Tensor output = stamp->getOutputTensor().clone();
    EXPECT_THROW(stamp->runOnTensorsForBenchmark(smaller, output, stream), std::invalid_argument);
    Tensor wrong_dtype = makeGpuTensor(values(257, rng), {257}, stream, DataType::FP16);
    EXPECT_THROW(stamp->runOnTensorsForBenchmark(input, wrong_dtype, stream), std::invalid_argument);

    CubArgReductionOutputOptions outputs;  // index only
    outputs.produce_value = false;
    outputs.index_output_dtype = DataType::UINT32;
    auto arg = CubArgReduction(CubArgReductionOp::ArgMin, 1U, outputs).stamp(input, stream);
    const std::optional<Tensor> index = arg->getIndexOutputTensor()->clone();
    const std::optional<Tensor> value = output;
    EXPECT_THROW(arg->runOnTensorsForBenchmark(input, value, index, stream), std::invalid_argument);
    EXPECT_THROW(arg->runOnTensorsForBenchmark(input, std::nullopt, std::nullopt, stream), std::invalid_argument);
}
