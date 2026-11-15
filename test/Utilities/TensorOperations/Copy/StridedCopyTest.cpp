// 1388: Thor's dense copy of a strided span (Utilities/TensorOperations/Copy/StridedCopy.h): the planner's families
// over canonical layouts (host only), and every family bit for bit against a host gather of the same bytes, across
// element widths, misaligned bases, partial and rectangular transpose tiles, batches, rebinding and indices past
// 2^31 words.

#include "Utilities/TensorOperations/Copy/StridedCopy.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

const TensorPlacement kGpu(TensorPlacement::MemDevices::GPU, 0);

void check(cudaError_t status, const char* what) { ASSERT_EQ(status, cudaSuccess) << what; }

uint32_t elementBytes(DataType dtype) { return static_cast<uint32_t>(TensorDescriptor::getElementSizeInBytes(dtype)); }

// A GPU tensor of `elements` elements whose bytes are a seeded sequence; `host` gets the same bytes.
Tensor filledTensor(DataType dtype, uint64_t elements, uint64_t seed, std::vector<uint8_t>& host, Stream& stream) {
    Tensor tensor(kGpu, TensorDescriptor(dtype, {elements}));
    host.resize(elements * elementBytes(dtype));
    std::mt19937_64 rng(seed);
    for (uint8_t& b : host) b = static_cast<uint8_t>(rng());
    check(cudaMemcpyAsync(tensor.getMemPtr<void>(), host.data(), host.size(), cudaMemcpyHostToDevice,
                          stream.getStream()),
          "upload");
    return tensor;
}

std::vector<uint8_t> download(const Tensor& tensor, Stream& stream) {
    std::vector<uint8_t> host(tensor.getArraySizeInBytes());
    check(cudaMemcpyAsync(host.data(), tensor.getMemPtr<void>(), host.size(), cudaMemcpyDeviceToHost,
                          stream.getStream()),
          "download");
    stream.synchronize();
    return host;
}

// The span's elements in row-major order of its dimensions, from the host copy of its storage.
std::vector<uint8_t> hostGather(const std::vector<uint8_t>& storage, uint64_t offset, const std::vector<uint64_t>& dims,
                                const std::vector<uint64_t>& strides, uint32_t e) {
    uint64_t total = 1;
    for (uint64_t d : dims) total *= d;
    std::vector<uint8_t> out(total * e);
    std::vector<uint64_t> c(dims.size(), 0);
    for (uint64_t j = 0; j < total; ++j) {
        uint64_t element = offset;
        for (size_t k = 0; k < dims.size(); ++k) element += c[k] * strides[k];
        std::memcpy(out.data() + j * e, storage.data() + element * e, e);
        for (size_t k = dims.size(); k-- > 0;) {
            if (++c[k] < dims[k]) break;
            c[k] = 0;
        }
    }
    return out;
}

uint64_t reach(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
    uint64_t r = 1;
    for (size_t k = 0; k < dims.size(); ++k) r += (dims[k] - 1) * strides[k];
    return r;
}

std::string describe(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides, DataType dtype,
                     uint64_t offset) {
    std::string s = "dims";
    for (uint64_t d : dims) s += " " + std::to_string(d);
    s += " strides";
    for (uint64_t x : strides) s += " " + std::to_string(x);
    return s + " bytes " + std::to_string(elementBytes(dtype)) + " offset " + std::to_string(offset);
}

// Copies the span (dims, strides) at element `offset` of a filled tensor and compares it with the host gather.
// Returns the plan.
DenseCopyPlan expectCopy(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides, DataType dtype,
                         uint64_t offset, Stream& stream, uint64_t seed = 1388) {
    SCOPED_TRACE(describe(dims, strides, dtype, offset));
    std::vector<uint8_t> host;
    const Tensor storage = filledTensor(dtype, offset + reach(dims, strides), seed, host, stream);
    const StridedSpan source(storage, offset, dims, strides);
    Tensor destination(kGpu, TensorDescriptor(dtype, dims));
    const DenseCopyPlan plan = planCopyToDense(source, destination);
    launchCopyToDense(plan, source, destination, stream);
    const std::vector<uint8_t> got = download(destination, stream);
    const std::vector<uint8_t> want = hostGather(host, offset, dims, strides, elementBytes(dtype));
    EXPECT_TRUE(got == want) << denseCopyPlanName(plan);
    return plan;
}

DenseCopyPlan hostPlan(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides, uint32_t e,
                       uintptr_t source = 0x10000, uintptr_t destination = 0x20000) {
    return DenseCopyInternal::planLayout(dims, strides, e, source, destination);
}

const std::vector<DataType> kWidths = {DataType::FP8_E4M3, DataType::FP16, DataType::FP32, DataType::UINT64};

}  // namespace

TEST(StridedCopy, PlansFollowTheCanonicalLayout) {
    // Dense spans, extent-one axes with any stride, a single element: one memcpy.
    EXPECT_EQ(hostPlan({4, 5}, {5, 1}, 4).family, DenseCopyFamily::Memcpy);
    EXPECT_EQ(hostPlan({1, 6, 1}, {99, 1, 7}, 2).family, DenseCopyFamily::Memcpy);
    EXPECT_EQ(hostPlan({1, 1}, {5, 9}, 1).family, DenseCopyFamily::Memcpy);
    EXPECT_EQ(denseCopyPlanName(hostPlan({4, 5}, {5, 1}, 4)), "memcpy_w4");

    // An F rotation: [b, a] of a dense [a, b]. x = the b axis (stride 1 = p), v = the a axis.
    DenseCopyPlan f = hostPlan({64, 48}, {1, 64}, 2);
    EXPECT_EQ(f.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(f.word_bytes, 2U);
    EXPECT_EQ(f.p, 1U);
    EXPECT_EQ(f.tile_x, 32U);
    EXPECT_EQ(f.tile_v, 32U);
    EXPECT_EQ(f.x, 64U);
    EXPECT_EQ(f.v, 48U);
    EXPECT_EQ(f.x_dest_stride, 48U);
    EXPECT_EQ(f.v_source_stride, 64U);
    EXPECT_EQ(denseCopyPlanName(f), "transpose_p1_w2_t32x32_b0");
    // The side grows with the vectors (about 512 tiles) up to a 128-byte tile row: 128 1-byte rows, 64 2-byte rows.
    EXPECT_EQ(hostPlan({64, 48}, {1, 64}, 1).tile_x, 32U);
    DenseCopyPlan f1 = hostPlan({4096, 4096}, {1, 4096}, 1);
    EXPECT_EQ(f1.tile_x, 128U);
    EXPECT_EQ(f1.tile_v, 128U);
    EXPECT_EQ(hostPlan({4096, 2048}, {1, 4096}, 2).tile_x, 64U);
    EXPECT_EQ(hostPlan({2, 4096, 4096}, {4096 * 4096, 1, 4096}, 4).tile_x, 32U);  // a batch axis: not a plain rotation

    // 1389e: a plain rotation of a dense matrix of single words goes to the matrix transpose kernel: at 4-byte words
    // whenever both sides are a tile (32) or more; at 2-byte words from 24 MiB of words, both sides 64 words or more
    // (1389k: 1-byte words never). Within its limits: under 2^31 words and at most 65,535 row tiles of v.
    DenseCopyPlan m = hostPlan({64, 48}, {1, 64}, 4);
    EXPECT_EQ(m.family, DenseCopyFamily::MatrixTranspose);
    EXPECT_EQ(m.word_bytes, 4U);
    EXPECT_EQ(m.x, 64U);
    EXPECT_EQ(m.v, 48U);
    EXPECT_EQ(m.x_dest_stride, 48U);
    EXPECT_EQ(m.v_source_stride, 64U);
    EXPECT_EQ(m.tile_x, 0U);
    EXPECT_EQ(DenseCopyInternal::workUnits(m), 0U);
    EXPECT_EQ(denseCopyPlanName(m), "matrix_transpose_w4");
    EXPECT_EQ(hostPlan({64, 31}, {1, 64}, 4).family, DenseCopyFamily::Transpose);            // a skinny side
    EXPECT_EQ(hostPlan({64, 48}, {1, 65}, 4).family, DenseCopyFamily::Transpose);            // a source with gaps
    EXPECT_EQ(hostPlan({64, 48}, {1, 64}, 8).family, DenseCopyFamily::Transpose);            // 8-byte words
    // 1389k: 1-byte words are Transposes (the packed byte tile) at any size; along v from x of half v.
    EXPECT_EQ(denseCopyPlanName(hostPlan({5123, 4999}, {1, 5123}, 1)), "transpose_p1_w1_t128x128_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({4000, 8000}, {1, 4000}, 1)), "transpose_p1_w1_t128x128_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({3999, 8000}, {1, 3999}, 1)), "transpose_p1_w1_t128x128_b0");  // under v / 2
    EXPECT_EQ(denseCopyPlanName(hostPlan({251, 100003}, {1, 251}, 1)), "transpose_p1_w1_t128x128_b0");
    EXPECT_FALSE(hostPlan({4000, 2000}, {1, 4000}, 2).tile_v_first);  // 2-byte words: from 8 v as before
    EXPECT_EQ(hostPlan({234227, 106}, {1, 234227}, 1).family, DenseCopyFamily::Transpose);        // v under 128
    EXPECT_EQ(denseCopyPlanName(hostPlan({4099, 3100}, {1, 4099}, 2)), "matrix_transpose_w2");   // 24.2 MiB
    EXPECT_EQ(hostPlan({134935, 92}, {1, 134935}, 2).family, DenseCopyFamily::Transpose);         // 23.7 MiB
    EXPECT_EQ(hostPlan({9, 3723975}, {1, 9}, 4).family, DenseCopyFamily::Transpose);              // skinny x
    EXPECT_EQ(hostPlan({32, 2097121}, {1, 32}, 4).family, DenseCopyFamily::Transpose);            // 65,536 row tiles
    EXPECT_EQ(hostPlan({32, 2097120}, {1, 32}, 4).family, DenseCopyFamily::MatrixTranspose);      // 65,535
    EXPECT_EQ(hostPlan({46341, 46341}, {1, 46341}, 4).family, DenseCopyFamily::Transpose);        // 2^31 words or more
    // 1389i: x of 8 v or more in 16 MiB of words or more is a Transpose with the tiles along v.
    EXPECT_EQ(denseCopyPlanName(hostPlan({16384, 256}, {1, 16384}, 4)), "transpose_p1_w4_t32x32_b0_vfirst");  // 16 MiB
    EXPECT_EQ(hostPlan({16384, 255}, {1, 16384}, 4).family, DenseCopyFamily::MatrixTranspose);    // under 16 MiB
    EXPECT_EQ(denseCopyPlanName(hostPlan({8000, 1000}, {1, 8000}, 4)), "transpose_p1_w4_t32x32_b0_vfirst");  // 8 v
    EXPECT_EQ(hostPlan({7999, 1000}, {1, 7999}, 4).family, DenseCopyFamily::MatrixTranspose);     // under 8 v
    EXPECT_EQ(denseCopyPlanName(hostPlan({96855, 2638}, {1, 96855}, 1)), "transpose_p1_w1_t128x128_b0_vfirst");
    // 1389s: rotations of 2- and 4-byte words with x of 7/8 v or more, from 48 MiB: the wide tile along v (256 x 16
    // 2-byte, 128 x 8 4-byte words), from the matrix kernel or the square tile; under either edge as before.
    EXPECT_EQ(denseCopyPlanName(hostPlan({278450, 198, 4}, {4, 1113800, 1}, 1)), "transpose_p1_w4_t128x8_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({19117, 5680}, {1, 19117}, 2)), "transpose_p1_w2_t256x16_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({13127, 4136}, {1, 13127}, 4)), "transpose_p1_w4_t128x8_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({41077, 1555}, {1, 41077}, 4)), "transpose_p1_w4_t128x8_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({3548, 3548}, {1, 3548}, 4)), "transpose_p1_w4_t128x8_b0_vfirst");  // 48 MiB
    EXPECT_EQ(denseCopyPlanName(hostPlan({3547, 3547}, {1, 3547}, 4)), "matrix_transpose_w4");
    EXPECT_EQ(denseCopyPlanName(hostPlan({7000, 8000}, {1, 7000}, 4)), "transpose_p1_w4_t128x8_b0_vfirst");  // 7/8 v
    EXPECT_EQ(denseCopyPlanName(hostPlan({6999, 8000}, {1, 6999}, 4)), "matrix_transpose_w4");
    EXPECT_EQ(denseCopyPlanName(hostPlan({5000, 4400}, {1, 5000}, 2)), "matrix_transpose_w2");  // 44 MB
    // Transposes: along v when x is 8 v or more and there are two tiles or more along each; else along x.
    EXPECT_TRUE(hostPlan({134935, 92}, {1, 134935}, 2).tile_v_first);
    EXPECT_FALSE(hostPlan({234227, 106}, {1, 234227}, 1).tile_v_first);  // v is one tile
    EXPECT_FALSE(hostPlan({34, 623019, 5}, {5, 170, 1}, 2).tile_v_first);  // x under v
    EXPECT_FALSE(hostPlan({1500, 1400, 3}, {3, 4500, 1}, 2).tile_v_first);  // x about v
    // 1389j: a tile of single bytes is packed: tile_x rows (rounded up to 4) of a power of two of 4-byte words.
    EXPECT_TRUE(DenseCopyInternal::transposePacksBytes(1, 1));
    EXPECT_FALSE(DenseCopyInternal::transposePacksBytes(3, 1));
    EXPECT_FALSE(DenseCopyInternal::transposePacksBytes(1, 2));
    EXPECT_EQ(DenseCopyInternal::transposeByteRowWords(128), 32U);
    EXPECT_EQ(DenseCopyInternal::transposeByteRowWords(131), 64U);
    EXPECT_EQ(DenseCopyInternal::transposeByteRowWords(9), 4U);
    EXPECT_EQ(DenseCopyInternal::transposeByteRowWords(1), 1U);
    EXPECT_EQ(DenseCopyInternal::sharedBytes(hostPlan({4096, 4096}, {1, 4096}, 1)), 128U * 32 * 4);
    EXPECT_EQ(DenseCopyInternal::transposeSharedBytes(125, 131, 1, 1), 128U * 64 * 4);  // the 32 KB of one tile
    EXPECT_EQ(DenseCopyInternal::transposeSharedBytes(1820, 9, 1, 1), 1820U * 4 * 4);
    EXPECT_EQ(DenseCopyInternal::transposeSharedBytes(32, 32, 1, 2), 32U * 34 * 2);  // 2-byte words: 17 banks
    // A side shorter than the tile is the whole side, and the other side grows to keep the square tile's vectors.
    DenseCopyPlan skinny_x = hostPlan({10, 3000}, {1, 10}, 4);
    EXPECT_EQ(skinny_x.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(skinny_x.tile_x, 10U);
    EXPECT_EQ(skinny_x.tile_v, 102U);
    DenseCopyPlan skinny_v = hostPlan({3000, 9}, {1, 3000}, 1);
    EXPECT_EQ(skinny_v.tile_x, 113U);
    EXPECT_EQ(skinny_v.tile_v, 9U);
    EXPECT_EQ(denseCopyPlanName(skinny_v), "transpose_p1_w1_t113x9_b0");
    // 1389l: a packed byte tile of 320 4 x 4 blocks or more (1389m; through 1389l 256) keeps its sides; else it grows.
    EXPECT_EQ(denseCopyPlanName(hostPlan({234227, 106}, {1, 234227}, 1)), "transpose_p1_w1_t128x106_b0");  // 154 x 106
    EXPECT_EQ(denseCopyPlanName(hostPlan({40, 300000}, {1, 40}, 1)), "transpose_p1_w1_t40x128_b0");  // 10 x 32 blocks
    EXPECT_EQ(denseCopyPlanName(hostPlan({36, 300000}, {1, 36}, 1)), "transpose_p1_w1_t36x455_b0");  // 9 x 32
    EXPECT_EQ(denseCopyPlanName(hostPlan({28, 300000}, {1, 28}, 1)), "transpose_p1_w1_t28x585_b0");  // 7 x 32
    EXPECT_EQ(denseCopyPlanName(hostPlan({51, 100000}, {1, 51}, 1)), "transpose_p1_w1_t51x99_b0");   // 51 x 192
    EXPECT_EQ(denseCopyPlanName(hostPlan({28, 300000}, {1, 28}, 2)), "transpose_p1_w2_t28x146_b0");  // 2-byte words
    // 1389t: a packed byte tile grown along x past four sides (512 bytes) that leaves under 2048 tiles stops there
    // (1820 x 9 in 656 tiles through 1389s); 2048 tiles or more keep the grown tile, and other words grow as before.
    EXPECT_EQ(denseCopyPlanName(hostPlan({1192132, 9}, {1, 1192132}, 1)), "transpose_p1_w1_t512x9_b0");
    EXPECT_EQ(DenseCopyInternal::workUnits(hostPlan({1192132, 9}, {1, 1192132}, 1)), 2329U);
    EXPECT_EQ(denseCopyPlanName(hostPlan({3725540, 9}, {1, 3725540}, 1)), "transpose_p1_w1_t512x9_b0");  // 2047
    EXPECT_EQ(denseCopyPlanName(hostPlan({3725541, 9}, {1, 3725541}, 1)), "transpose_p1_w1_t1820x9_b0");  // 2048
    EXPECT_EQ(denseCopyPlanName(hostPlan({9, 184, 18, 27, 9}, {804816, 486, 27, 1, 89424}, 1)),
              "transpose_p1_w1_t512x9_b1");  // 1547 x 9 under 9 batch elements through 1389s
    EXPECT_EQ(denseCopyPlanName(hostPlan({51, 65, 66, 9, 13}, {501930, 7722, 9, 1, 594}, 1)),
              "transpose_p1_w1_t594x13_b1");  // 3315 tiles
    EXPECT_EQ(denseCopyPlanName(hostPlan({400001, 9}, {1, 400001}, 2)), "transpose_p1_w2_t455x9_b0");

    // G rotations: [b, a, p] of a dense [a, b, p], rows widened to the word the bytes allow.
    DenseCopyPlan g2 = hostPlan({64, 48, 2}, {2, 128, 1}, 2);  // 4-byte rows: a plain rotation of 4-byte words
    EXPECT_EQ(g2.family, DenseCopyFamily::MatrixTranspose);
    EXPECT_EQ(g2.word_bytes, 4U);
    EXPECT_EQ(g2.p, 1U);
    EXPECT_EQ(hostPlan({64, 31, 2}, {2, 128, 1}, 2).family, DenseCopyFamily::Transpose);
    DenseCopyPlan g7 = hostPlan({40, 30, 7}, {7, 280, 1}, 4);  // 28-byte rows: 4-byte words
    EXPECT_EQ(g7.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(g7.word_bytes, 4U);
    EXPECT_EQ(g7.p, 7U);
    EXPECT_EQ(g7.tile_x, 13U);  // 1389d: the smallest square tile of 1,024 words or more (13 x 13 x 7 = 1,183)
    EXPECT_EQ(g7.tile_v, 13U);
    // 1389d: the smallest side counts words (through 1389c: 32 vectors, so 1,024 p words a tile). A short x is still
    // the whole side, with v grown to keep 13 x 13 vectors; about 512 tiles or more and the side grows as before.
    EXPECT_EQ(denseCopyPlanName(hostPlan({10, 1134, 7}, {7, 70, 1}, 4)), "transpose_p7_w4_t10x16_b0");
    EXPECT_EQ(DenseCopyInternal::workUnits(hostPlan({10, 1134, 7}, {7, 70, 1}, 4)), 71U);  // 12 through 1389c
    EXPECT_EQ(denseCopyPlanName(hostPlan({1768, 113, 3}, {3, 5304, 1}, 4)), "transpose_p3_w4_t19x19_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(hostPlan({58, 5196, 3}, {3, 174, 1}, 4)), "transpose_p3_w4_t24x24_b0");
    EXPECT_EQ(denseCopyPlanName(hostPlan({32, 337530, 5}, {5, 160, 1}, 4)), "transpose_p5_w4_t32x32_b0");
    EXPECT_EQ(denseCopyPlanName(hostPlan({4464, 179, 3}, {3, 13392, 1}, 1)), "transpose_p3_w1_t39x39_b0_vfirst");
    // 1389r: unpacked single bytes (1-byte words, p of 3, 5, 7) from 16 MiB take tile rows of 32 bytes (7 vectors
    // of 5) and v rows to 32 KB; under 16 MiB and at 2-byte words the square tile as before.
    EXPECT_EQ(denseCopyPlanName(hostPlan({37, 1149980, 5}, {5, 185, 1}, 1)), "transpose_p5_w1_t7x853_b0");
    EXPECT_EQ(DenseCopyInternal::workUnits(hostPlan({37, 1149980, 5}, {5, 185, 1}, 1)), 8094U);  // 71874 at 32 x 32
    EXPECT_EQ(denseCopyPlanName(hostPlan({37, 90688, 5}, {5, 185, 1}, 1)), "transpose_p5_w1_t7x850_b0");  // 16 MiB
    EXPECT_EQ(denseCopyPlanName(hostPlan({37, 90687, 5}, {5, 185, 1}, 1)), "transpose_p5_w1_t32x32_b0");
    EXPECT_EQ(denseCopyPlanName(hostPlan({37, 1149980, 5}, {5, 185, 1}, 2)), "transpose_p5_w2_t32x32_b0");
    EXPECT_EQ(denseCopyPlanName(hostPlan({3000, 2000, 3}, {3, 9000, 1}, 1)), "transpose_p3_w1_t11x900_b0");
    DenseCopyPlan g8 = hostPlan({40, 30, 8}, {8, 320, 1}, 4);  // a 32-byte row under 4 MiB: copied in order
    EXPECT_EQ(g8.family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(g8.word_bytes, 16U);
    EXPECT_EQ(g8.p, 2U);
    EXPECT_EQ(g8.outer_dims, (std::vector<uint64_t>{40, 30}));
    EXPECT_EQ(g8.outer_strides, (std::vector<uint64_t>{2, 80}));

    // 1389f: rows of two to four 16-byte words (32, 48, 64 bytes) are transposed in a copy of 4 MiB or more.
    DenseCopyPlan wide = hostPlan({256, 512, 8}, {8, 2048, 1}, 4);  // 4 MiB of 32-byte rows
    EXPECT_EQ(wide.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(wide.word_bytes, 16U);
    EXPECT_EQ(wide.p, 2U);
    EXPECT_EQ(wide.x, 256U);
    EXPECT_EQ(wide.v, 512U);
    EXPECT_EQ(wide.x_dest_stride, 1024U);  // 512 rows of 2 words
    EXPECT_EQ(wide.v_source_stride, 512U);
    EXPECT_EQ(denseCopyPlanName(wide), "transpose_p2_w16_t23x23_b0");
    EXPECT_LE(DenseCopyInternal::sharedBytes(wide), DenseCopyInternal::TRANSPOSE_MAX_SHARED_BYTES);
    EXPECT_EQ(hostPlan({256, 511, 8}, {8, 2048, 1}, 4).family, DenseCopyFamily::RowCopy);        // under 4 MiB
    EXPECT_EQ(denseCopyPlanName(hostPlan({946, 2485, 16}, {16, 15136, 1}, 4)), "transpose_p4_w16_t22x22_b0");
    EXPECT_EQ(denseCopyPlanName(hostPlan({160, 32620, 12}, {12, 1920, 1}, 4)), "transpose_p3_w16_t25x25_b0");
    EXPECT_EQ(denseCopyPlanName(hostPlan({1296, 3629, 16}, {16, 20736, 1}, 2)), "transpose_p2_w16_t32x28_b0");
    for (const DenseCopyPlan& plan : {hostPlan({946, 2485, 16}, {16, 15136, 1}, 4),
                                      hostPlan({160, 32620, 12}, {12, 1920, 1}, 4),
                                      hostPlan({1296, 3629, 16}, {16, 20736, 1}, 2)})
        EXPECT_LE(DenseCopyInternal::sharedBytes(plan), DenseCopyInternal::TRANSPOSE_MAX_SHARED_BYTES);
    EXPECT_EQ(hostPlan({946, 2485, 20}, {20, 18920, 1}, 4).family, DenseCopyFamily::RowCopy);  // 80-byte rows
    EXPECT_EQ(hostPlan({946, 2485, 9}, {9, 8514, 1}, 4).family, DenseCopyFamily::RowCopy);     // 36 bytes: 4-byte words
    EXPECT_EQ(hostPlan({946, 2485, 16}, {16, 15136, 1}, 4, 0x10004).family, DenseCopyFamily::RowCopy);  // unaligned
    EXPECT_EQ(hostPlan({946, 2485, 16}, {16, 15137, 1}, 4).family, DenseCopyFamily::RowCopy);  // 4-byte words too
    DenseCopyPlan wide_heads = hostPlan({34, 13202, 12, 8}, {1267392, 8, 105616, 1}, 4);  // [B, S, H, 8 FP32]
    EXPECT_EQ(wide_heads.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(wide_heads.batch_dims, (std::vector<uint64_t>{34}));
    EXPECT_EQ(wide_heads.batch_source, (std::vector<uint64_t>{316848}));  // in 16-byte words
    EXPECT_EQ(wide_heads.batch_dest, (std::vector<uint64_t>{316848}));
    EXPECT_EQ(denseCopyPlanName(wide_heads), "transpose_p2_w16_t75x12_b1");

    // H packings: rows of 6 FP16 elements, 10 apart, slabs 100 apart; rows of 6, 10 apart (one pitch).
    DenseCopyPlan h = hostPlan({3, 4, 6}, {100, 10, 1}, 2);
    EXPECT_EQ(h.family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(h.word_bytes, 4U);
    EXPECT_EQ(h.p, 3U);
    EXPECT_EQ(h.outer_strides, (std::vector<uint64_t>{50, 5}));
    EXPECT_EQ(denseCopyPlanName(h), "row_copy_o2_p3_w4");
    DenseCopyPlan rows = hostPlan({12, 6}, {10, 1}, 2);
    EXPECT_EQ(rows.family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(rows.outer_dims, (std::vector<uint64_t>{12}));
    // Merged axes: a slab pitch of R x Sr is one pitch.
    EXPECT_EQ(hostPlan({3, 4, 6}, {40, 10, 1}, 2).outer_dims, (std::vector<uint64_t>{12}));
    // A misaligned base keeps the element width.
    EXPECT_EQ(hostPlan({12, 6}, {10, 1}, 2, 0x10002).word_bytes, 2U);
    EXPECT_EQ(hostPlan({12, 6}, {10, 1}, 2, 0x10000, 0x20002).word_bytes, 2U);

    // Attention head permutes [B, S, H, D] of [B, H, S, D]: D = 64 FP16 rows are 128 bytes (row copy); D = 8 rows
    // are 16 bytes, so S (stride D) is x, H is v and B a batch axis.
    EXPECT_EQ(hostPlan({2, 64, 12, 64}, {12 * 64 * 64, 64, 64 * 64, 1}, 2).family, DenseCopyFamily::RowCopy);
    DenseCopyPlan heads = hostPlan({2, 64, 12, 8}, {12 * 64 * 8, 8, 64 * 8, 1}, 2);
    EXPECT_EQ(heads.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(heads.word_bytes, 16U);
    EXPECT_EQ(heads.x, 64U);
    EXPECT_EQ(heads.v, 12U);
    EXPECT_EQ(heads.tile_x, 64U);
    EXPECT_EQ(heads.tile_v, 12U);
    EXPECT_EQ(heads.batch_dims, (std::vector<uint64_t>{2}));
    EXPECT_EQ(heads.batch_source, (std::vector<uint64_t>{12 * 64}));  // in 16-byte words
    EXPECT_EQ(heads.batch_dest, (std::vector<uint64_t>{64 * 12}));
    EXPECT_EQ(denseCopyPlanName(heads), "transpose_p1_w16_t64x12_b1");
    // Three batch axes, outer to inner: [3, 2, 4, 30, 20] of a dense [2, 3, 4, 20, 30].
    DenseCopyPlan batches = hostPlan({3, 2, 4, 30, 20}, {2400, 7200, 600, 1, 30}, 2);
    EXPECT_EQ(batches.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(batches.batch_dims, (std::vector<uint64_t>{3, 2, 4}));
    EXPECT_EQ(batches.batch_source, (std::vector<uint64_t>{2400, 7200, 600}));
    EXPECT_EQ(batches.batch_dest, (std::vector<uint64_t>{4800, 2400, 600}));
    EXPECT_EQ(denseCopyPlanName(batches), "transpose_p1_w2_t30x20_b3");

    // An x by v plane under 256 vectors: the row copy (the plane is a few sectors), or for single words of 4 bytes
    // or fewer the packed gather (1389b). 1389d: the plane, since a tile of 7-word vectors is 169 of them.
    EXPECT_EQ(hostPlan({2, 16, 12, 8}, {12 * 16 * 8, 8, 16 * 8, 1}, 2).family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(hostPlan({4, 48}, {1, 4}, 4).family, DenseCopyFamily::PackedGather);
    EXPECT_EQ(hostPlan({48, 4}, {1, 48}, 4).family, DenseCopyFamily::PackedGather);
    EXPECT_EQ(hostPlan({33, 9}, {1, 33}, 4).family, DenseCopyFamily::Transpose);  // 297 vectors
    EXPECT_EQ(hostPlan({15, 17, 7}, {7, 105, 1}, 4).family, DenseCopyFamily::RowCopy);    // 255 vectors of 7 words
    EXPECT_EQ(hostPlan({16, 16, 7}, {7, 112, 1}, 4).family, DenseCopyFamily::Transpose);  // 256
    // 1389a: more than 2,048 tiles of under 640 words each are not a transpose. [B, 4, x, 25] of a dense
    // [B, 4, 25, x] (B and 4 merge into one batch axis): 375 words a plane at x = 15, 650 at x = 26.
    const auto planes = [](uint64_t b, uint64_t x, uint32_t e) {
        return hostPlan({b, 4, x, 25}, {100 * x, 25 * x, 1, x}, e);
    };
    EXPECT_EQ(planes(512, 15, 1).family, DenseCopyFamily::Transpose);  // 2,048 tiles of 375 words
    EXPECT_EQ(denseCopyPlanName(planes(512, 15, 1)), "transpose_p1_w1_t15x25_b1");
    EXPECT_EQ(planes(513, 15, 1).family, DenseCopyFamily::PackedGather);  // 2,052 tiles
    EXPECT_EQ(planes(513, 15, 1).outer_dims, (std::vector<uint64_t>{2052, 15, 25}));
    EXPECT_EQ(planes(513, 15, 8).family, DenseCopyFamily::RowCopy);    // words, whatever their width
    EXPECT_EQ(planes(513, 26, 1).family, DenseCopyFamily::Transpose);  // 650 words a tile
    EXPECT_EQ(std::string(denseCopyRevision()), "1395");  // 1395: the planner's row-copy counts only
    // 1389h: the grid is one CTA per unit of work, 0 for a plan with none.
    EXPECT_EQ(DenseCopyInternal::gridOf(planes(512, 15, 1)), 2048U);
    EXPECT_EQ(DenseCopyInternal::gridOf(planes(513, 15, 1)), 188U);
    EXPECT_EQ(DenseCopyInternal::gridOf(hostPlan({70001, 31}, {41, 1}, 4)), 2120U);  // 1389p: 1024 words a step
    // A row copy's words per thread (rowCopyStepItems, 1389p): one 4-byte word or two 2-byte words up to 768 KiB, from
    // 16 MiB and on single words over a sector apart, else 16 bytes a thread; 16 single bytes, four words of 8 or 16
    // bytes. 1395: the only counts (1389n-1394: a plan's row_items could set another).
    const DenseCopyPlan rows33 = hostPlan({40001}, {33}, 4);
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(rows33), 1U);
    EXPECT_EQ(DenseCopyInternal::workUnits(rows33), 157U);
    const DenseCopyPlan rows31 = hostPlan({7001, 31}, {41, 1}, 2);
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(rows31), 2U);
    EXPECT_EQ(DenseCopyInternal::workUnits(rows31), 424U);  // 217,031 words, 512 a step
    const DenseCopyPlan bytes12 = hostPlan({70, 104, 12}, {3640, 35, 1}, 1);
    EXPECT_EQ(denseCopyPlanName(bytes12), "row_copy_o1_p12_w1");
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(bytes12), 16U);
    const DenseCopyPlan words8 = planes(513, 15, 8);
    EXPECT_EQ(words8.family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(words8.word_bytes, 8U);
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(words8), 4U);
    EXPECT_EQ(DenseCopyInternal::rowCopyItems(16), 4U);
    EXPECT_EQ(DenseCopyInternal::rowCopyShortItems(4), 1U);
    EXPECT_EQ(DenseCopyInternal::rowCopyShortItems(2), 2U);
    EXPECT_EQ(DenseCopyInternal::rowCopyShortItems(1), 16U);
    // 1389p: the short step by destination size, word and stride.
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({70001, 31}, {41, 1}, 2)), 8U);    // 4.3 MB: long
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({300001, 31}, {41, 1}, 2)), 2U);   // 18.6 MB: short
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({196608}, {2}, 4)), 1U);            // 768 KiB: short
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({196609}, {2}, 4)), 4U);
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({4194304}, {2}, 4)), 1U);           // 16 MiB: short
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({4194303}, {2}, 4)), 4U);
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({1000001}, {33}, 4)), 1U);          // 132 bytes apart
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({1000001}, {8}, 4)), 4U);           // 32 bytes apart
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({2000001}, {17}, 2)), 2U);          // 34 bytes apart
    EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(hostPlan({2000001}, {16}, 2)), 8U);
    EXPECT_EQ(DenseCopyInternal::transposeLoadItems(1), 16U);  // the transpose's loads in flight: as before
    EXPECT_EQ(DenseCopyInternal::transposeLoadItems(2), 8U);
    EXPECT_EQ(DenseCopyInternal::transposeLoadItems(4), 4U);
    EXPECT_EQ(DenseCopyInternal::transposeLoadItems(16), 4U);
    EXPECT_EQ(DenseCopyInternal::gridOf(hostPlan({1000, 77}, {1, 1000}, 4)), 0U);  // the matrix kernel
    EXPECT_EQ(DenseCopyInternal::gridOf(hostPlan({1000}, {1}, 4)), 0U);            // one memcpy
    // 1389g: the plan's name marks the tiles along v (1389i: the planner's when x is 8 v or more).
    DenseCopyPlan ordered = planes(512, 15, 1);
    EXPECT_FALSE(ordered.tile_v_first);
    ordered.tile_v_first = true;
    EXPECT_EQ(denseCopyPlanName(ordered), "transpose_p1_w1_t15x25_b1_vfirst");
    // 1389b: the packed gather is single words of 1, 2 or 4 bytes over two or more outer axes into a 16-byte aligned
    // destination; 8-byte words, one outer axis and other destinations are row copies (2^31 words or more too:
    // IndicesPast2To31Words).
    const DenseCopyPlan packed = planes(513, 15, 1);
    EXPECT_EQ(packed.p, 1U);
    EXPECT_EQ(packed.outer_strides, (std::vector<uint64_t>{375, 1, 15}));
    EXPECT_EQ(denseCopyPlanName(packed), "packed_gather_o3_w1");
    EXPECT_EQ(DenseCopyInternal::workUnits(packed), 188U);  // 48,093 packs of 16 bytes, 256 a step
    EXPECT_EQ(denseCopyPlanName(planes(513, 15, 2)), "packed_gather_o3_w2");
    EXPECT_EQ(denseCopyPlanName(planes(513, 15, 4)), "packed_gather_o3_w4");
    EXPECT_EQ(denseCopyPlanName(hostPlan({513, 4, 15, 25}, {1500, 375, 1, 15}, 1, 0x10000, 0x20008)),
              "row_copy_o3_p1_w1");
    EXPECT_EQ(DenseCopyInternal::workUnits(hostPlan({3, 2}, {1, 3}, 1)), 1U);  // 6 words: no whole pack, one step
    // A broadcast v axis (stride 0) still tiles; a strided rank-1 view is a gather.
    EXPECT_EQ(hostPlan({64, 48}, {1, 0}, 4).family, DenseCopyFamily::Transpose);
    DenseCopyPlan gather = hostPlan({100}, {3}, 4);
    EXPECT_EQ(gather.family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(gather.p, 1U);
    EXPECT_EQ(gather.outer_strides, (std::vector<uint64_t>{3}));
    // A rank-6 reversal: no axis merges and none is contiguous, so six outer axes of single words.
    DenseCopyPlan reversed = hostPlan({2, 3, 2, 3, 2, 3}, {1, 2, 6, 12, 36, 72}, 4);
    EXPECT_EQ(reversed.family, DenseCopyFamily::PackedGather);
    EXPECT_EQ(reversed.outer_dims.size(), 6U);
    // A rank-6 permutation whose two innermost axes merge.
    DenseCopyPlan merged = hostPlan({3, 2, 3, 2, 3, 2}, {72, 1, 36, 2, 12, 6}, 4);
    EXPECT_EQ(merged.family, DenseCopyFamily::PackedGather);
    EXPECT_EQ(merged.outer_dims, (std::vector<uint64_t>{3, 2, 3, 2, 6}));

    EXPECT_THROW(hostPlan({}, {}, 4), std::invalid_argument);
    EXPECT_THROW(hostPlan(std::vector<uint64_t>(33, 1), std::vector<uint64_t>(33, 1), 4), std::invalid_argument);
    EXPECT_THROW(hostPlan({4}, {1}, 3), std::invalid_argument);
    EXPECT_THROW(hostPlan({4, 2}, {1}, 4), std::invalid_argument);
}

TEST(StridedCopy, CopiesEveryFamilyBitForBit) {
    Stream stream(kGpu);
    struct Layout {
        std::vector<uint64_t> dims, strides;
    };
    const std::vector<Layout> layouts = {
        {{257, 129}, {1, 257}},                     // F rotation, partial tiles on both sides
        {{33, 9}, {1, 33}},                         // just above the smallest plane (256 vectors)
        {{10, 3000}, {1, 10}},                      // 1389: a skinny x side, partial v tiles
        {{3000, 9}, {1, 3000}},                     // 1389: a skinny v side, partial x tiles
        {{3, 2, 4, 30, 20}, {2400, 7200, 600, 1, 30}},  // 1389: three batch axes
        {{520, 4, 15, 25}, {1500, 375, 1, 15}},     // 1389a: 2,080 planes of 375 words: a packed gather (1389b)
        {{5, 4, 6, 3, 7}, {504, 3, 12, 1, 72}},     // 1389b: five outer axes, carries past both stepped axes
        {{7, 3, 5}, {1, 35, 7}},                    // 1389b: 105 words: a tail after the last whole pack
        {{3, 2}, {1, 3}},                           // 1389b: 6 words: at 1 and 2 bytes only the tail
        {{130, 65, 3}, {3, 390, 1}},                // G, 3-element rows (1389d: tiles of 19 x 19 at 4 bytes)
        {{65, 31, 7}, {7, 455, 1}},                 // G, 7-element rows (1389d: 13 x 13, partial v tiles)
        {{10, 300, 7}, {7, 70, 1}},                 // 1389d: G, a skinny x side of 7-element rows (10 x 16)
        {{40, 30, 9}, {9, 360, 1}},                 // G, rows past a sector for 4-byte elements
        {{3, 129, 33, 4}, {33 * 129 * 4, 4, 129 * 4, 1}},  // head permute, one batch axis
        {{2, 3, 70, 20, 2}, {2800, 5600, 2, 140, 1}},  // two batch axes (of a dense [3, 2, 20, 70, 2])
        {{3, 4, 6}, {100, 10, 1}},                  // H, both gaps
        {{12, 5}, {7, 1}},                          // H, row gaps
        {{2, 9, 7}, {200, 7, 1}},                   // H, slab gaps
        {{1000}, {3}},                              // rank-1 gather
        {{64, 48}, {1, 0}},                         // broadcast v axis
        {{48, 64}, {0, 1}},                         // broadcast rows
        {{3, 2, 3, 2, 3, 2}, {72, 1, 36, 2, 12, 6}},  // rank-6 permutation
        {{5, 6}, {6, 1}},                           // dense
    };
    for (DataType dtype : kWidths)
        for (const Layout& layout : layouts)
            for (uint64_t offset : {0ULL, 1ULL, 3ULL}) expectCopy(layout.dims, layout.strides, dtype, offset, stream);
}

// 1389e: plain rotations of a dense matrix run the matrix transpose kernel: 4-byte words at any size (partial tiles on
// both sides; 4-byte rows of two FP16 elements), 2-byte words from 24 MiB, on bases aligned to its packed 4-byte
// words and not. 1389k: 1-byte words are Transposes (the packed byte tile), here the rotation it took through 1389j.
TEST(StridedCopy, PlainRotationsRunTheMatrixKernel) {
    Stream stream(kGpu);
    for (uint64_t offset : {0ULL, 4ULL}) {
        EXPECT_EQ(denseCopyPlanName(expectCopy({257, 129}, {1, 257}, DataType::FP32, offset, stream)),
                  "matrix_transpose_w4");
        EXPECT_EQ(denseCopyPlanName(expectCopy({130, 65, 2}, {2, 260, 1}, DataType::FP16, offset, stream)),
                  "matrix_transpose_w4");
    }
    for (uint64_t offset : {0ULL, 1ULL}) {
        EXPECT_EQ(denseCopyPlanName(expectCopy({5123, 4999}, {1, 5123}, DataType::FP8_E4M3, offset, stream)),
                  "transpose_p1_w1_t128x128_b0_vfirst");
        EXPECT_EQ(denseCopyPlanName(expectCopy({4099, 3100}, {1, 4099}, DataType::FP16, offset, stream)),
                  "matrix_transpose_w2");
    }
    // 1389s: from 48 MiB with x of 7/8 v or more, the wide tile along v (partial tiles on both sides).
    for (uint64_t offset : {0ULL, 2ULL})
        EXPECT_EQ(denseCopyPlanName(expectCopy({5003, 5101}, {1, 5003}, DataType::FP16, offset, stream)),
                  "transpose_p1_w2_t256x16_b0_vfirst");
    EXPECT_EQ(denseCopyPlanName(expectCopy({3555, 3549}, {1, 3555}, DataType::FP32, 0, stream)),
              "transpose_p1_w4_t128x8_b0_vfirst");
}

// 1389f: rows of two to four 16-byte words in a copy of 4 MiB or more go through the transpose tile: partial tiles on
// both sides, a batch axis, a base 16 bytes on; a base that is not 16-byte aligned keeps the row copy.
TEST(StridedCopy, WideRowsTransposeFromFourMiB) {
    Stream stream(kGpu);
    for (uint64_t offset : {0ULL, 4ULL})
        EXPECT_EQ(denseCopyPlanName(expectCopy({1030, 257, 8}, {8, 8240, 1}, DataType::FP32, offset, stream)),
                  "transpose_p2_w16_t23x23_b0");
    EXPECT_EQ(expectCopy({1030, 257, 8}, {8, 8240, 1}, DataType::FP32, 1, stream).family, DenseCopyFamily::RowCopy);
    for (uint64_t offset : {0ULL, 8ULL})
        EXPECT_EQ(denseCopyPlanName(expectCopy({700, 130, 24}, {24, 16800, 1}, DataType::FP16, offset, stream)),
                  "transpose_p3_w16_t19x19_b0");
    for (uint64_t offset : {0ULL, 16ULL})
        EXPECT_EQ(denseCopyPlanName(expectCopy({5, 1200, 12, 64}, {12 * 1200 * 64, 64, 1200 * 64, 1},
                                               DataType::FP8_E4M3, offset, stream)),
                  "transpose_p4_w16_t21x12_b1");
    EXPECT_EQ(denseCopyPlanName(expectCopy({300, 1100, 16}, {16, 4800, 1}, DataType::FP32, 0, stream)),
              "transpose_p4_w16_t22x22_b0");
}

// 1389r: unpacked single bytes from 16 MiB on the tall tile, with partial tiles along x (37 = 5 x 7 + 2, 3000 = 272
// x 11 + 8) and along v, bit for bit, at an aligned and an odd byte offset.
TEST(StridedCopy, UnpackedByteVectorsTakeTallTiles) {
    Stream stream(kGpu);
    for (uint64_t offset : {0ULL, 1ULL}) {
        EXPECT_EQ(denseCopyPlanName(expectCopy({37, 100000, 5}, {5, 185, 1}, DataType::FP8_E4M3, offset, stream)),
                  "transpose_p5_w1_t7x821_b0");
        EXPECT_EQ(denseCopyPlanName(expectCopy({3000, 2000, 3}, {3, 9000, 1}, DataType::FP8_E5M2, offset, stream)),
                  "transpose_p3_w1_t11x900_b0");
    }
}

// 1389g: the transpose takes its tiles in either order, and a plan whose tile was changed by hand (the audit's probe
// does both), with that tile's grid, copies the same bytes: tall, wide and small tiles with partial tiles on both
// sides, rows of three words, a batch axis, and a matrix transpose's plan run as a tile transpose.
TEST(StridedCopy, EitherTileOrderAndHandTilesCopyTheSame) {
    Stream stream(kGpu);
    struct Layout {
        std::vector<uint64_t> dims, strides;
        DataType dtype;
    };
    const std::vector<Layout> layouts = {
        {{1000, 77}, {1, 1000}, DataType::FP32},                       // [x, v] of a dense [v, x]: the matrix kernel
        {{333, 50, 3}, {3, 999, 1}, DataType::FP16},                   // rows of three 2-byte words
        {{3, 150, 2, 41}, {12300, 1, 6150, 150}, DataType::FP8_E4M3},  // a batch axis; v is 2 x 41
    };
    const std::vector<std::pair<uint32_t, uint32_t>> tiles = {{32, 32}, {3, 50}, {64, 5}, {7, 7}, {1, 41}, {150, 1}};
    for (const Layout& layout : layouts) {
        SCOPED_TRACE(describe(layout.dims, layout.strides, layout.dtype, 0));
        std::vector<uint8_t> host;
        const Tensor storage = filledTensor(layout.dtype, reach(layout.dims, layout.strides), 1389, host, stream);
        const StridedSpan source(storage, 0, layout.dims, layout.strides);
        Tensor destination(kGpu, TensorDescriptor(layout.dtype, layout.dims));
        const std::vector<uint8_t> want =
            hostGather(host, 0, layout.dims, layout.strides, elementBytes(layout.dtype));
        DenseCopyPlan plan = planCopyToDense(source, destination);
        EXPECT_EQ(plan.family == DenseCopyFamily::MatrixTranspose, layout.dims.size() == 2);
        plan.family = DenseCopyFamily::Transpose;  // a matrix transpose's plan has the transpose's x, v and strides
        for (const auto& [tile_x, tile_v] : tiles) {
            for (bool v_first : {false, true}) {
                plan.tile_x = tile_x;
                plan.tile_v = tile_v;
                plan.tile_v_first = v_first;
                plan.grid = DenseCopyInternal::gridOf(plan);
                ASSERT_GE(plan.grid, 1U);
                check(cudaMemsetAsync(destination.getMemPtr<void>(), 0, destination.getArraySizeInBytes(),
                                      stream.getStream()),
                      "clear");
                launchCopyToDense(plan, source, destination, stream);
                EXPECT_TRUE(download(destination, stream) == want) << denseCopyPlanName(plan);
            }
        }
        plan.grid = 3;  // fewer CTAs than tiles, each taking every third tile
        launchCopyToDense(plan, source, destination, stream);
        EXPECT_TRUE(download(destination, stream) == want) << denseCopyPlanName(plan) << " grid 3";
    }
}

// 1395: the row copy at each count the planner takes (each word width, the short and the long step of 2- and 4-byte
// words), on the plan's grid and on 3 CTAs, bit for bit. Through 1394 this ran every count of 1, 2, 4, 8 and 16 the
// audit's probe measured (RowCopiesTakeAnyWordsPerThread); only the planner's are compiled now.
TEST(StridedCopy, RowCopiesRunThePlannersWordsPerThread) {
    Stream stream(kGpu);
    struct Layout {
        std::vector<uint64_t> dims, strides;
        DataType dtype;
        uint32_t word_bytes, items;
    };
    const std::vector<Layout> layouts = {
        {{40001}, {33}, DataType::FP32, 4, 1},           // single words over a sector apart: the short step
        {{70001, 31}, {41, 1}, DataType::FP32, 4, 4},    // 8.7 MB: the long step
        {{7001, 31}, {41, 1}, DataType::FP16, 2, 2},     // 434 KB: the short step
        {{70001, 31}, {41, 1}, DataType::FP16, 2, 8},    // 4.3 MB: the long step
        {{70, 104, 12}, {3640, 35, 1}, DataType::FP8_E4M3, 1, 16},
        {{20001, 4}, {6, 1}, DataType::FP32, 8, 4},      // rows of 16 bytes 24 bytes apart: 8-byte words
        {{20001, 4}, {8, 1}, DataType::FP32, 16, 4},     // 32 bytes apart: 16-byte words
    };
    for (const Layout& layout : layouts) {
        SCOPED_TRACE(describe(layout.dims, layout.strides, layout.dtype, 0));
        std::vector<uint8_t> host;
        const Tensor storage = filledTensor(layout.dtype, reach(layout.dims, layout.strides), 1395, host, stream);
        const StridedSpan source(storage, 0, layout.dims, layout.strides);
        Tensor destination(kGpu, TensorDescriptor(layout.dtype, layout.dims));
        const std::vector<uint8_t> want =
            hostGather(host, 0, layout.dims, layout.strides, elementBytes(layout.dtype));
        DenseCopyPlan plan = planCopyToDense(source, destination);
        ASSERT_EQ(plan.family, DenseCopyFamily::RowCopy);
        EXPECT_EQ(plan.word_bytes, layout.word_bytes);
        EXPECT_EQ(DenseCopyInternal::rowCopyStepItems(plan), layout.items);
        for (const uint32_t grid : {plan.grid, 3U}) {  // 3: fewer CTAs than steps, each taking every third step
            plan.grid = grid;
            check(cudaMemsetAsync(destination.getMemPtr<void>(), 0, destination.getArraySizeInBytes(),
                                  stream.getStream()),
                  "clear");
            launchCopyToDense(plan, source, destination, stream);
            EXPECT_TRUE(download(destination, stream) == want) << denseCopyPlanName(plan) << " grid " << grid;
        }
    }
}

// 1389h: a copy launches one CTA per unit of work. A row copy, a packed gather and a transpose of more units than
// CTAs are resident at once (through 1389g the grid was the resident CTAs: 510 to 1,020 on 170 SMs), bit for bit.
TEST(StridedCopy, LargeCopiesLaunchOneCtaPerUnit) {
    Stream stream(kGpu);
    const DenseCopyPlan rows = expectCopy({70001, 31}, {41, 1}, DataType::FP32, 0, stream);
    EXPECT_EQ(rows.family, DenseCopyFamily::RowCopy);
    EXPECT_EQ(rows.grid, 2120U);  // 1389p: four words a thread at 8.7 MB (1389o: one, 8477 CTAs)
    const DenseCopyPlan packed = expectCopy({60, 300, 400}, {2, 120, 36000}, DataType::FP8_E4M3, 0, stream);
    EXPECT_EQ(packed.family, DenseCopyFamily::PackedGather);
    EXPECT_EQ(packed.grid, 1758U);
    const DenseCopyPlan tiles = expectCopy({1500, 1400, 3}, {3, 4500, 1}, DataType::FP16, 0, stream);
    EXPECT_EQ(tiles.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(tiles.grid, DenseCopyInternal::workUnits(tiles));
    EXPECT_EQ(tiles.grid, 2068U);
}

// 1389i: rotations of long source rows take the tiles along v, the 16 MiB ones off the matrix kernel, bit for bit.
TEST(StridedCopy, LongSourceRowsTakeTilesAlongV) {
    Stream stream(kGpu);
    const DenseCopyPlan rotation = expectCopy({16384, 256}, {1, 16384}, DataType::FP32, 0, stream);
    EXPECT_EQ(denseCopyPlanName(rotation), "transpose_p1_w4_t32x32_b0_vfirst");
    EXPECT_EQ(rotation.grid, 4096U);
    const DenseCopyPlan bytes = expectCopy({131072, 256}, {1, 131072}, DataType::FP8_E4M3, 0, stream);
    EXPECT_EQ(denseCopyPlanName(bytes), "transpose_p1_w1_t128x128_b0_vfirst");
    const DenseCopyPlan rows = expectCopy({134935, 92}, {1, 134935}, DataType::FP16, 0, stream);
    EXPECT_EQ(denseCopyPlanName(rows), "transpose_p1_w2_t64x64_b0_vfirst");
    const DenseCopyPlan small = expectCopy({16384, 255}, {1, 16384}, DataType::FP32, 0, stream);
    EXPECT_EQ(small.family, DenseCopyFamily::MatrixTranspose);
    const DenseCopyPlan groups = expectCopy({1768, 113, 3}, {3, 5304, 1}, DataType::FP32, 0, stream);
    EXPECT_EQ(denseCopyPlanName(groups), "transpose_p3_w4_t19x19_b0_vfirst");
}

// 1389j: transposes of single bytes go four bytes at a time from the aligned words that hold them: every base offset
// and odd strides, partial tiles on both sides, a skinny x, a batch axis, tiles along v, 128 x 128 tiles, bit for bit.
TEST(StridedCopy, SingleByteTransposesPackFourBytes) {
    Stream stream(kGpu);
    struct Layout {
        std::vector<uint64_t> dims, strides;
    };
    const std::vector<Layout> layouts = {
        {{1301, 1103}, {1, 1301}},                         // odd rows, partial tiles
        {{2638, 905}, {1, 2638}},                          // rows of 2 mod 4 bytes
        {{11001, 9}, {1, 11001}},                          // v in one tile of 9 rows
        {{10, 40001}, {1, 10}},                            // a skinny x
        {{51, 7, 9, 180}, {1, 51 * 9 * 180, 51, 51 * 9}},  // a batch axis
        {{129, 1000}, {1, 131}},                           // a source with gaps
        {{50001, 301}, {1, 50001}},                        // x of 8 v or more: tiles along v
        {{8193, 2051}, {1, 8193}},                         // 128 x 128 tiles along v (1389k), 16 MiB
        {{51, 100000}, {1, 51}},                           // 1389l: a short x, the tile not grown (51 x 99)
        {{36, 300000}, {1, 36}},                           // 1389m: 288 blocks, grown (36 x 455)
        {{400001, 9}, {1, 400001}},                        // 1389t: x grown to 512 bytes at most (782 tiles)
        {{7, 60001, 9}, {540009, 1, 60001}},               // 1389t: the same under 7 batch elements (512 x 9)
    };
    for (const Layout& layout : layouts) {
        for (uint64_t offset : {0ULL, 1ULL, 2ULL, 3ULL}) {
            const DenseCopyPlan plan = expectCopy(layout.dims, layout.strides, DataType::FP8_E4M3, offset, stream);
            EXPECT_EQ(plan.family, DenseCopyFamily::Transpose) << denseCopyPlanName(plan);
            EXPECT_EQ(plan.word_bytes, 1U);
            EXPECT_EQ(plan.p, 1U);
        }
    }
    EXPECT_EQ(denseCopyPlanName(expectCopy({8193, 2051}, {1, 8193}, DataType::FP8_E4M3, 0, stream)),
              "transpose_p1_w1_t128x128_b0_vfirst");  // 1389k: x of half v or more
    EXPECT_TRUE(expectCopy({50001, 301}, {1, 50001}, DataType::FP8_E4M3, 1, stream).tile_v_first);
    EXPECT_EQ(denseCopyPlanName(expectCopy({400001, 9}, {1, 400001}, DataType::FP8_E4M3, 3, stream)),
              "transpose_p1_w1_t512x9_b0");  // 1389t: 765 x 9 through 1389s
}

TEST(StridedCopy, CopiesSeededRandomLayouts) {
    Stream stream(kGpu);
    std::mt19937_64 rng(1388);
    const uint64_t extents[] = {1, 2, 3, 5, 8, 9, 17, 32, 33, 64, 65, 129};
    for (int iteration = 0; iteration < 160; ++iteration) {
        const DataType dtype = kWidths[rng() % kWidths.size()];
        const size_t rank = 1 + rng() % 5;
        std::vector<uint64_t> base(rank), physical(rank), physical_strides(rank);
        uint64_t elements = 1;
        for (size_t k = 0; k < rank; ++k) {
            base[k] = extents[rng() % std::size(extents)];
            while (elements * base[k] > 300000 && base[k] > 1) base[k] /= 2;
            elements *= base[k];
            physical[k] = base[k] + (rng() % 4 == 0 ? 1 + rng() % 5 : 0);  // gaps
        }
        uint64_t stride = 1;
        for (size_t k = rank; k-- > 0;) {
            physical_strides[k] = stride;
            stride *= physical[k];
        }
        std::vector<size_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<uint64_t> dims(rank), strides(rank);
        for (size_t k = 0; k < rank; ++k) {
            dims[k] = base[order[k]];
            strides[k] = physical_strides[order[k]];
        }
        if (rng() % 8 == 0) strides[rng() % rank] = 0;
        expectCopy(dims, strides, dtype, rng() % 4, stream, iteration);
        if (HasFailure()) return;
    }
}

TEST(StridedCopy, StampedPlanRunsOnReboundSpans) {
    Stream stream(kGpu);
    const std::vector<uint64_t> dims = {130, 31, 2}, strides = {2, 260, 1};  // 31: a tile transpose (1389e)
    std::vector<uint8_t> host;
    const uint64_t span = reach(dims, strides);
    const Tensor storage = filledTensor(DataType::FP16, 2 * span + 8, 7, host, stream);
    Tensor first(kGpu, TensorDescriptor(DataType::FP16, dims)), second(kGpu, TensorDescriptor(DataType::FP16, dims));
    const StridedSpan a(storage, 0, dims, strides);
    const DenseCopyPlan plan = planCopyToDense(a, first);
    EXPECT_EQ(plan.family, DenseCopyFamily::Transpose);
    EXPECT_EQ(plan.word_bytes, 4U);
    const uint64_t offset = span + 6;  // 12 bytes on: aligned to the 4-byte word
    const StridedSpan b(storage, offset, dims, strides);
    launchCopyToDense(plan, a, first, stream);
    launchCopyToDense(plan, b, second, stream);
    EXPECT_TRUE(download(first, stream) == hostGather(host, 0, dims, strides, 2));
    EXPECT_TRUE(download(second, stream) == hostGather(host, offset, dims, strides, 2));

    // Another alignment, layout or dtype is refused.
    EXPECT_THROW(launchCopyToDense(plan, StridedSpan(storage, 1, dims, strides), first, stream), std::invalid_argument);
    EXPECT_THROW(launchCopyToDense(plan, StridedSpan(storage, 0, {65, 130, 2}, {260, 2, 1}),
                                   Tensor(kGpu, TensorDescriptor(DataType::FP16, {65, 130, 2})), stream),
                 std::invalid_argument);
    Tensor wrong_dtype(kGpu, TensorDescriptor(DataType::BF16, dims));
    EXPECT_THROW(launchCopyToDense(plan, a, wrong_dtype, stream), std::invalid_argument);
    EXPECT_THROW((void)planCopyToDense(a, wrong_dtype), std::invalid_argument);
    Tensor wrong_shape(kGpu, TensorDescriptor(DataType::FP16, {65, 130, 2}));
    EXPECT_THROW(copyToDenseAsync(a, wrong_shape, stream), std::invalid_argument);

    // 1389b: a packed gather stores 16-byte words, so its plan is refused on a dense destination that is not 16-byte
    // aligned; that destination's own plan is the row copy.
    const std::vector<uint64_t> pdims = {7, 3, 5}, pstrides = {1, 35, 7};
    const StridedSpan permuted(storage, 0, pdims, pstrides);
    Tensor aligned(kGpu, TensorDescriptor(DataType::FP16, pdims));
    Tensor roomy(kGpu, TensorDescriptor(DataType::FP16, {106}));
    check(cudaMemsetAsync(roomy.getMemPtr<void>(), 0, roomy.getArraySizeInBytes(), stream.getStream()), "clear");
    const StridedSpan shifted(roomy, 1, pdims, {15, 5, 1});  // dense, 2 bytes into its tensor
    const DenseCopyPlan packed_plan = planCopyToDense(permuted, aligned);
    EXPECT_EQ(packed_plan.family, DenseCopyFamily::PackedGather);
    EXPECT_THROW(launchCopyToDense(packed_plan, permuted, shifted, stream), std::invalid_argument);
    const DenseCopyPlan shifted_plan = planCopyToDense(permuted, shifted);
    EXPECT_EQ(shifted_plan.family, DenseCopyFamily::RowCopy);
    launchCopyToDense(packed_plan, permuted, aligned, stream);
    launchCopyToDense(shifted_plan, permuted, shifted, stream);
    const std::vector<uint8_t> want = hostGather(host, 0, pdims, pstrides, 2);
    EXPECT_TRUE(download(aligned, stream) == want);
    const std::vector<uint8_t> shifted_bytes = download(roomy, stream);
    EXPECT_TRUE(std::equal(want.begin(), want.end(), shifted_bytes.begin() + 2));
    EXPECT_EQ(shifted_bytes[0], 0);
    EXPECT_EQ(shifted_bytes[1], 0);
}

// Destinations of more than 2^31 one-byte words: the 64-bit row copy and the transpose's 64-bit offsets. The sources
// broadcast an axis, so they stay small; sampled chunks of the 2 GiB destinations are checked.
TEST(StridedCopy, IndicesPast2To31Words) {
    Stream stream(kGpu);
    const uint64_t n = (uint64_t{1} << 20) + 3;
    struct Case {
        std::vector<uint64_t> dims, strides;
        DenseCopyFamily family;
    };
    for (const Case& c : {Case{{2048, n}, {0, 3}, DenseCopyFamily::RowCopy},
                          Case{{n + 2, 2049}, {1, 0}, DenseCopyFamily::Transpose}}) {
        SCOPED_TRACE(describe(c.dims, c.strides, DataType::UINT8, 0));
        std::vector<uint8_t> host;
        const Tensor storage = filledTensor(DataType::UINT8, reach(c.dims, c.strides), 9, host, stream);
        const StridedSpan source(storage, 0, c.dims, c.strides);
        Tensor destination(kGpu, TensorDescriptor(DataType::UINT8, c.dims));
        const DenseCopyPlan plan = planCopyToDense(source, destination);
        ASSERT_EQ(plan.family, c.family);
        ASSERT_GE(plan.words, uint64_t{1} << 31);
        launchCopyToDense(plan, source, destination, stream);
        stream.synchronize();
        const uint64_t total = c.dims[0] * c.dims[1];
        constexpr uint64_t chunk = 4096;
        std::vector<uint8_t> got(chunk);
        for (uint64_t s = 0; s <= 64; ++s) {
            const uint64_t start = std::min(total - chunk, (total / 64) * s + 4093 * s);
            check(cudaMemcpyAsync(got.data(), destination.getMemPtr<uint8_t>() + start, chunk, cudaMemcpyDeviceToHost,
                                  stream.getStream()),
                  "sample");
            stream.synchronize();
            for (uint64_t k = 0; k < chunk; ++k) {
                const uint64_t j = start + k;
                const uint64_t i0 = j / c.dims[1], i1 = j % c.dims[1];
                const uint8_t want = host[i0 * c.strides[0] + i1 * c.strides[1]];
                if (got[k] != want) {
                    ADD_FAILURE() << "word " << j << ": " << int(got[k]) << " against " << int(want);
                    return;
                }
            }
        }
    }
}
