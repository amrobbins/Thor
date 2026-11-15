// 1379: the RK, KR and R families cover every site of every floating input dtype at every element alignment of its
// base. A dense span need not start on a 16-byte boundary (a span of a larger storage, or a view's bytes read as the
// dense tensor they are), so each cell stamps and runs one input at each alignment its dtype allows: FP32 4, 8 and 16
// bytes; FP16 / BF16 2, 4, 8 and 16; FP8 E4M3 / E5M2 1, 2, 4, 8 and 16. The shapes cross the limits the families have:
// R around the rotated passes' 32-row shard (2, 8, 32, 33, 97), K below, at and past a packet (1, 2, 3, 5, 8, 16, 17,
// 130), KR rows, pure R, and two reduced sites with short R (the first pass of those reads the base, the next an FP32
// intermediate). Every cell runs VALUE SUM, MAX and MEAN against a host reference (small integers, so SUM and MAX are
// exact in any order) and ARGMAX (the value is the maximum and the index names an element holding it). Tuning depth
// Quick. Every failure is collected and printed, then the test fails if there was any.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Internal = ThorImplementation::CubReductionInternal;

namespace {

struct Shape {
    std::vector<uint64_t> dims;
    std::vector<uint32_t> axes;
};

std::string describe(const Shape& shape) {
    std::ostringstream out;
    for (size_t i = 0; i < shape.dims.size(); ++i) out << (i ? "x" : "") << shape.dims[i];
    out << " over {";
    for (size_t i = 0; i < shape.axes.size(); ++i) out << (i ? "," : "") << shape.axes[i];
    out << '}';
    return out.str();
}

std::vector<Shape> shapes() {
    std::vector<Shape> all;
    for (const uint64_t r : {2, 8, 32, 33, 97})
        for (const uint64_t k : {1, 2, 3, 5, 8, 16, 17, 130}) all.push_back({{3, r, k}, {1}});
    for (const uint64_t r : {3, 31, 100, 5000}) all.push_back({{5, r}, {1}});  // KR
    for (const uint64_t r : {7, 1000, 100000}) all.push_back({{r}, {0}});      // R
    all.push_back({{3, 8, 4, 5, 6}, {1, 3}});
    all.push_back({{2, 11, 3, 8, 5}, {1, 3}});
    all.push_back({{3, 40, 2, 3, 17}, {1, 3}});
    all.push_back({{2, 2, 3, 64, 2}, {1, 3}});
    return all;
}

uint64_t elementBytes(DataType dtype) {
    return dtype == DataType::FP32 ? 4 : (dtype == DataType::FP16 || dtype == DataType::BF16) ? 2 : 1;
}

const char* dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return "fp32";
        case DataType::FP16: return "fp16";
        case DataType::BF16: return "bf16";
        case DataType::FP8_E4M3: return "e4m3";
        case DataType::FP8_E5M2: return "e5m2";
        default: return "?";
    }
}

// For each element (row-major), its output (retained coordinates, row-major) and its index in the reduction domain
// (reduced coordinates, row-major over the sorted axes).
struct Layout {
    std::vector<uint64_t> output, reduced;
    uint64_t outputs = 1, count = 1;
};

Layout layoutOf(const Shape& shape) {
    std::vector<bool> is_reduced(shape.dims.size(), false);
    for (uint32_t axis : shape.axes) is_reduced[axis] = true;
    Layout layout;
    for (size_t a = 0; a < shape.dims.size(); ++a) (is_reduced[a] ? layout.count : layout.outputs) *= shape.dims[a];
    uint64_t elements = layout.outputs * layout.count;
    layout.output.resize(elements);
    layout.reduced.resize(elements);
    std::vector<uint64_t> index(shape.dims.size(), 0);
    for (uint64_t e = 0; e < elements; ++e) {
        uint64_t o = 0, r = 0;
        for (size_t a = 0; a < shape.dims.size(); ++a) {
            if (is_reduced[a]) r = r * shape.dims[a] + index[a];
            else o = o * shape.dims[a] + index[a];
        }
        layout.output[e] = o;
        layout.reduced[e] = r;
        for (size_t a = shape.dims.size(); a-- > 0;) {
            if (++index[a] < shape.dims[a]) break;
            index[a] = 0;
        }
    }
    return layout;
}

// `values` as `dtype` elements `offset_bytes` into a UINT8 storage kept alive in `storages`.
StridedSpan atOffset(const std::vector<float>& values, const std::vector<uint64_t>& dims, DataType dtype,
                     uint64_t offset_bytes, std::vector<Tensor>& storages, Stream& stream) {
    const uint64_t bytes = values.size() * elementBytes(dtype);
    const Tensor packed = makeGpuTensor(values, {values.size()}, stream, dtype);
    storages.emplace_back(gpuPlacement, TensorDescriptor(DataType::UINT8, {offset_bytes + bytes + 16}));
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(static_cast<uint8_t*>(storages.back().getMemPtr<void>()) + offset_bytes,
                                           packed.getMemPtr<void>(), bytes, cudaMemcpyDeviceToDevice,
                                           stream.getStream()));
    stream.synchronize();
    return StridedSpan(storages.back()).bytesAs(offset_bytes, dtype, dims);
}

}  // namespace

TEST(CubReductionAlignmentCoverage, EveryRKKRAndRSiteReducesAtEveryElementAlignment) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    std::mt19937_64 rng(1379);
    const std::vector<Shape> all_shapes = shapes();
    std::vector<std::string> failures;
    std::map<std::string, uint64_t> failures_by_group;
    uint64_t cells = 0;
    for (const DataType dtype :
         {DataType::FP32, DataType::FP16, DataType::BF16, DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const uint64_t alignment : {uint64_t{1}, uint64_t{2}, uint64_t{4}, uint64_t{8}, uint64_t{16}}) {
            if (alignment < elementBytes(dtype)) continue;
            const uint64_t offset_bytes = alignment == 16 ? 0 : alignment;
            for (const Shape& shape : all_shapes) {
                const Layout layout = layoutOf(shape);
                std::vector<float> values(layout.output.size());
                for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 7) - 3);
                std::vector<Tensor> storages;
                const StridedSpan input = atOffset(values, shape.dims, dtype, offset_bytes, storages, stream);
                const std::string cell = std::string(dtypeName(dtype)) + " align " + std::to_string(alignment) + " " +
                                         describe(shape);
                const auto fail = [&](const std::string& what, const std::string& op) {
                    failures.push_back(cell + " " + op + ": " + what);
                    ++failures_by_group[std::string(dtypeName(dtype)) + " align " + std::to_string(alignment)];
                };
                // Host references.
                std::vector<double> sum(layout.outputs, 0.0), maximum(layout.outputs, -1e30);
                for (uint64_t e = 0; e < values.size(); ++e) {
                    sum[layout.output[e]] += values[e];
                    maximum[layout.output[e]] = std::max<double>(maximum[layout.output[e]], values[e]);
                }
                for (const CubReductionOp op : {CubReductionOp::Sum, CubReductionOp::Max, CubReductionOp::Mean}) {
                    const std::string name = op == CubReductionOp::Sum ? "sum" : op == CubReductionOp::Max ? "max"
                                                                                                           : "mean";
                    ++cells;
                    try {
                        const CubReduction reduction(op, shape.axes, DataType::FP32, 1.0f,
                                                     CubReductionTuningDepth::Quick);
                        auto stamped = reduction.stamp(input, stream);
                        stamped->run();
                        stream.synchronize();
                        const std::vector<float> got = copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
                        if (got.size() != layout.outputs) {
                            fail("output size " + std::to_string(got.size()), name);
                            continue;
                        }
                        for (uint64_t o = 0; o < layout.outputs; ++o) {
                            const double mean = sum[o] / static_cast<double>(layout.count);
                            const double want = op == CubReductionOp::Sum   ? sum[o]
                                                : op == CubReductionOp::Max ? maximum[o]
                                                                            : mean;
                            const double tolerance = op == CubReductionOp::Mean ? 1e-6 * std::max(1.0, std::abs(want))
                                                                                : 0.0;
                            if (!(std::abs(got[o] - want) <= tolerance)) {
                                std::ostringstream what;
                                what << "output " << o << " got " << got[o] << " want " << want << " ("
                                     << stamped->getModernRKStrategyChain() << ")";
                                fail(what.str(), name);
                                break;
                            }
                        }
                    } catch (const std::exception& e) {
                        fail(e.what(), name);
                    }
                }
                ++cells;
                try {
                    CubArgReductionOutputOptions outputs;
                    outputs.value_output_dtype = DataType::FP32;
                    outputs.index_output_dtype = DataType::UINT64;
                    auto arg = CubArgReduction(CubArgReductionOp::ArgMax, shape.axes, outputs,
                                               CubReductionTuningDepth::Quick)
                                   .stamp(input, stream);
                    arg->run();
                    stream.synchronize();
                    const std::vector<float> value = copyGpuTensorAsFloat(*arg->getValueOutputTensor(), stream);
                    const std::vector<uint64_t> index = copyGpuTensorAsUnsigned(*arg->getIndexOutputTensor(), stream);
                    // The element of each (output, reduced index).
                    std::vector<float> at(layout.outputs * layout.count);
                    for (uint64_t e = 0; e < values.size(); ++e)
                        at[layout.output[e] * layout.count + layout.reduced[e]] = values[e];
                    for (uint64_t o = 0; o < layout.outputs; ++o) {
                        if (value[o] != maximum[o] || index[o] >= layout.count ||
                            at[o * layout.count + index[o]] != maximum[o]) {
                            std::ostringstream what;
                            what << "output " << o << " value " << value[o] << " index " << index[o] << " want "
                                 << maximum[o];
                            fail(what.str(), "argmax");
                            break;
                        }
                    }
                } catch (const std::exception& e) {
                    fail(e.what(), "argmax");
                }
            }
        }
    }
    std::ostringstream summary;
    summary << "[ alignment coverage ] cells=" << cells << " failures=" << failures.size();
    for (const auto& [group, count] : failures_by_group) summary << " | " << group << ": " << count;
    std::cout << summary.str() << std::endl;
    for (size_t i = 0; i < failures.size() && i < 80; ++i) std::cout << "  " << failures[i] << std::endl;
    EXPECT_TRUE(failures.empty()) << failures.size() << " cells failed (listed above)";
}
