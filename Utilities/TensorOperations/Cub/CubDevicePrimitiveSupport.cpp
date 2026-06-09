#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include <algorithm>
#include <limits>

namespace ThorImplementation::CubDevicePrimitiveSupport {

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void requireDenseContiguousGpuTensor(const Tensor& tensor, const char* name) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string(name) + " must be initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(name) + " must be a GPU tensor.");
    }
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument(std::string(name) + " must be dense contiguous.");
    }
}

void requireSameGpuPlacement(const Tensor& a, const Tensor& b, const char* a_name, const char* b_name) {
    if (a.getPlacement() != b.getPlacement()) {
        throw std::invalid_argument(std::string(a_name) + " and " + b_name + " must live on the same GPU placement.");
    }
}

void requireStorageForNumItems(const Tensor& tensor, const char* name, uint64_t num_items) {
    if (tensor.getTotalNumElements() < num_items) {
        throw std::invalid_argument(std::string(name) + " has fewer elements than num_items.");
    }
}

int checkedCubNumItems(uint64_t num_items) {
    if (num_items > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUB device primitive num_items exceeds CUB's int item-count limit.");
    }
    return static_cast<int>(num_items);
}

int64_t checkedCubInt64Count(uint64_t count, const char* name) {
    if (count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument(std::string(name) + " exceeds CUB's int64 count limit.");
    }
    return static_cast<int64_t>(count);
}

int fullEndBitFor(DataType dtype, int end_bit) {
    if (end_bit != 0) {
        return end_bit;
    }
    return static_cast<int>(TensorDescriptor::getElementSizeInBytes(dtype) * 8);
}

void validateBitRange(DataType dtype, int begin_bit, int end_bit) {
    const int full_bits = static_cast<int>(TensorDescriptor::getElementSizeInBytes(dtype) * 8);
    if (begin_bit < 0 || end_bit <= begin_bit || end_bit > full_bits) {
        throw std::invalid_argument("Invalid CUB radix-sort bit range for dtype " + dtypeName(dtype) + ".");
    }
}

void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes) {
    requireDenseContiguousGpuTensor(temp_storage, "temp_storage");
    if (temp_storage.getPlacement() != placement) {
        throw std::invalid_argument("temp_storage must live on the same GPU placement as the operation tensors.");
    }
    if (temp_storage.getDataType() != DataType::UINT8) {
        throw std::invalid_argument("temp_storage must have dtype uint8.");
    }
    if (temp_storage.getArraySizeInBytes() < temp_storage_bytes) {
        throw std::invalid_argument("temp_storage has fewer bytes than temp_storage_bytes.");
    }
}

void* mutableCubTempStoragePtr(const Tensor& temp_storage) {
    return const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
}

void validateSortKeys(const Tensor& keys_in, const Tensor& keys_out, uint64_t num_items, int begin_bit, int end_bit) {
    requireDenseContiguousGpuTensor(keys_in, "keys_in");
    requireDenseContiguousGpuTensor(keys_out, "keys_out");
    requireSameGpuPlacement(keys_in, keys_out, "keys_in", "keys_out");
    requireStorageForNumItems(keys_in, "keys_in", num_items);
    requireStorageForNumItems(keys_out, "keys_out", num_items);
    if (keys_in.getDataType() != keys_out.getDataType()) {
        throw std::invalid_argument("CUB radix-sort key input/output dtypes must match.");
    }
    if (!isCubRadixSortKeyDTypeSupported(keys_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB radix-sort key dtype " + dtypeName(keys_in.getDataType()) + ".");
    }
    validateBitRange(keys_in.getDataType(), begin_bit, end_bit);
}

void validateSortPairs(const Tensor& keys_in,
                       const Tensor& keys_out,
                       const Tensor& values_in,
                       const Tensor& values_out,
                       uint64_t num_items,
                       int begin_bit,
                       int end_bit) {
    validateSortKeys(keys_in, keys_out, num_items, begin_bit, end_bit);
    requireDenseContiguousGpuTensor(values_in, "values_in");
    requireDenseContiguousGpuTensor(values_out, "values_out");
    requireSameGpuPlacement(keys_in, values_in, "keys_in", "values_in");
    requireSameGpuPlacement(keys_in, values_out, "keys_in", "values_out");
    requireStorageForNumItems(values_in, "values_in", num_items);
    requireStorageForNumItems(values_out, "values_out", num_items);
    if (values_in.getDataType() != values_out.getDataType()) {
        throw std::invalid_argument("CUB radix-sort value input/output dtypes must match.");
    }
    if (!isCubRadixSortValueDTypeSupported(values_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB radix-sort index value dtype " + dtypeName(values_in.getDataType()) + ".");
    }
}

void validateTopKKeys(const Tensor& keys_in, const Tensor& keys_out, uint64_t num_items, uint64_t k) {
    requireDenseContiguousGpuTensor(keys_in, "keys_in");
    requireDenseContiguousGpuTensor(keys_out, "keys_out");
    requireSameGpuPlacement(keys_in, keys_out, "keys_in", "keys_out");
    requireStorageForNumItems(keys_in, "keys_in", num_items);
    requireStorageForNumItems(keys_out, "keys_out", std::min(num_items, k));
    if (keys_in.getDataType() != keys_out.getDataType()) {
        throw std::invalid_argument("CUB top-k key input/output dtypes must match.");
    }
    if (!isCubTopKKeyDTypeSupported(keys_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB top-k key dtype " + dtypeName(keys_in.getDataType()) + ".");
    }
    static_cast<void>(checkedCubInt64Count(num_items, "num_items"));
    static_cast<void>(checkedCubInt64Count(k, "k"));
}

void validateTopKPairs(const Tensor& keys_in,
                       const Tensor& keys_out,
                       const Tensor& values_in,
                       const Tensor& values_out,
                       uint64_t num_items,
                       uint64_t k) {
    validateTopKKeys(keys_in, keys_out, num_items, k);
    requireDenseContiguousGpuTensor(values_in, "values_in");
    requireDenseContiguousGpuTensor(values_out, "values_out");
    requireSameGpuPlacement(keys_in, values_in, "keys_in", "values_in");
    requireSameGpuPlacement(keys_in, values_out, "keys_in", "values_out");
    requireStorageForNumItems(values_in, "values_in", num_items);
    requireStorageForNumItems(values_out, "values_out", std::min(num_items, k));
    if (values_in.getDataType() != values_out.getDataType()) {
        throw std::invalid_argument("CUB top-k value input/output dtypes must match.");
    }
    if (!isCubTopKValueDTypeSupported(values_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB top-k index value dtype " + dtypeName(values_in.getDataType()) + ".");
    }
}

void validateSelectFlagged(const Tensor& input,
                           const Tensor& flags,
                           const Tensor& output,
                           const Tensor& num_selected_out,
                           uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(flags, "flags");
    requireDenseContiguousGpuTensor(output, "output");
    requireDenseContiguousGpuTensor(num_selected_out, "num_selected_out");
    requireSameGpuPlacement(input, flags, "input", "flags");
    requireSameGpuPlacement(input, output, "input", "output");
    requireSameGpuPlacement(input, num_selected_out, "input", "num_selected_out");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(flags, "flags", num_items);
    requireStorageForNumItems(output, "output", num_items);
    requireStorageForNumItems(num_selected_out, "num_selected_out", 1);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument("CUB select input/output dtypes must match.");
    }
    if (!isCubSelectValueDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB select value dtype " + dtypeName(input.getDataType()) + ".");
    }
    if (!isCubSelectFlagDTypeSupported(flags.getDataType())) {
        throw std::invalid_argument("Unsupported CUB select flag dtype " + dtypeName(flags.getDataType()) + ".");
    }
    if (num_selected_out.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB select num_selected_out must have dtype uint32.");
    }
    static_cast<void>(checkedCubNumItems(num_items));
}

void validateFindBounds(const Tensor& range,
                        const Tensor& values,
                        const Tensor& output,
                        uint64_t range_num_items,
                        uint64_t values_num_items,
                        const char* op_name) {
    requireDenseContiguousGpuTensor(range, "range");
    requireDenseContiguousGpuTensor(values, "values");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(range, values, "range", "values");
    requireSameGpuPlacement(range, output, "range", "output");
    requireStorageForNumItems(range, "range", range_num_items);
    requireStorageForNumItems(values, "values", values_num_items);
    requireStorageForNumItems(output, "output", values_num_items);
    if (range.getDataType() != values.getDataType()) {
        throw std::invalid_argument(std::string("CUB device ") + op_name + " range/value dtypes must match.");
    }
    if (!isCubFindKeyDTypeSupported(range.getDataType())) {
        throw std::invalid_argument("Unsupported CUB find key dtype " + dtypeName(range.getDataType()) + ".");
    }
    if (output.getDataType() != DataType::UINT32) {
        throw std::invalid_argument(std::string("CUB device ") + op_name + " output must have dtype uint32.");
    }
    static_cast<void>(checkedCubNumItems(range_num_items));
    static_cast<void>(checkedCubNumItems(values_num_items));
}

void validateFindIfFlagged(const Tensor& flags, const Tensor& index_out, uint64_t num_items) {
    requireDenseContiguousGpuTensor(flags, "flags");
    requireDenseContiguousGpuTensor(index_out, "index_out");
    requireSameGpuPlacement(flags, index_out, "flags", "index_out");
    requireStorageForNumItems(flags, "flags", num_items);
    requireStorageForNumItems(index_out, "index_out", 1);
    if (!isCubFindFlagDTypeSupported(flags.getDataType())) {
        throw std::invalid_argument("Unsupported CUB find-if flag dtype " + dtypeName(flags.getDataType()) + ".");
    }
    if (index_out.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB find-if index_out must have dtype uint32.");
    }
    static_cast<void>(checkedCubNumItems(num_items));
}

namespace {

// Keep this in sync with CubTopK.cu. CUB's fixed-size segmented top-k
// dispatch encodes the maximum segment size as a compile-time parameter and
// rejects bounds larger than the selected worker tile size.
constexpr uint64_t kSegmentedTopKMaxSegmentSize = 8192ULL;
constexpr uint64_t kSegmentedTopKMaxK = 8192ULL;

}  // namespace

uint64_t checkedSegmentedTopKTotalItems(uint64_t num_segments, uint64_t segment_size) {
    if (segment_size != 0 && num_segments > std::numeric_limits<uint64_t>::max() / segment_size) {
        throw std::invalid_argument("CUB segmented top-k num_segments * segment_size overflows uint64_t.");
    }
    const uint64_t total_items = num_segments * segment_size;
    if (total_items > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("CUB segmented top-k total item count exceeds the current CUB fixed-size implementation limit.");
    }
    static_cast<void>(checkedCubInt64Count(total_items, "num_segments * segment_size"));
    return total_items;
}

uint64_t checkedSegmentedTopKOutputItems(uint64_t num_segments, uint64_t segment_size, uint64_t k) {
    const uint64_t selected_per_segment = std::min(segment_size, k);
    if (selected_per_segment != 0 && num_segments > std::numeric_limits<uint64_t>::max() / selected_per_segment) {
        throw std::invalid_argument("CUB segmented top-k num_segments * k overflows uint64_t.");
    }
    const uint64_t output_items = num_segments * selected_per_segment;
    static_cast<void>(checkedCubInt64Count(output_items, "num_segments * k"));
    return output_items;
}

void validateSegmentedTopKKeys(const Tensor& keys_in,
                               const Tensor& keys_out,
                               uint64_t num_segments,
                               uint64_t segment_size,
                               uint64_t k) {
    requireDenseContiguousGpuTensor(keys_in, "keys_in");
    requireDenseContiguousGpuTensor(keys_out, "keys_out");
    requireSameGpuPlacement(keys_in, keys_out, "keys_in", "keys_out");

    const uint64_t total_items = checkedSegmentedTopKTotalItems(num_segments, segment_size);
    const uint64_t output_items = checkedSegmentedTopKOutputItems(num_segments, segment_size, k);
    requireStorageForNumItems(keys_in, "keys_in", total_items);
    requireStorageForNumItems(keys_out, "keys_out", output_items);

    if (keys_in.getDataType() != keys_out.getDataType()) {
        throw std::invalid_argument("CUB segmented top-k key input/output dtypes must match.");
    }
    if (!isCubTopKKeyDTypeSupported(keys_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented top-k key dtype " + dtypeName(keys_in.getDataType()) + ".");
    }

    static_cast<void>(checkedCubInt64Count(num_segments, "num_segments"));
    static_cast<void>(checkedCubInt64Count(segment_size, "segment_size"));
    static_cast<void>(checkedCubInt64Count(k, "k"));

    if (num_segments == 0 || k == 0) {
        return;
    }
    if (segment_size == 0) {
        throw std::invalid_argument("CUB segmented top-k segment_size must be nonzero when num_segments and k are nonzero.");
    }
    if (segment_size > kSegmentedTopKMaxSegmentSize) {
        throw std::invalid_argument("CUB segmented top-k segment_size exceeds CUB's current fixed-size support limit.");
    }
    if (k > kSegmentedTopKMaxK) {
        throw std::invalid_argument("CUB segmented top-k k exceeds CUB's current fixed-size support limit.");
    }
    if (k >= segment_size) {
        throw std::invalid_argument("CUB segmented top-k currently requires k to be smaller than the fixed segment size.");
    }
}

void validateSegmentedTopKPairs(const Tensor& keys_in,
                                const Tensor& keys_out,
                                const Tensor& values_in,
                                const Tensor& values_out,
                                uint64_t num_segments,
                                uint64_t segment_size,
                                uint64_t k) {
    validateSegmentedTopKKeys(keys_in, keys_out, num_segments, segment_size, k);
    const uint64_t total_items = checkedSegmentedTopKTotalItems(num_segments, segment_size);
    const uint64_t output_items = checkedSegmentedTopKOutputItems(num_segments, segment_size, k);

    requireDenseContiguousGpuTensor(values_in, "values_in");
    requireDenseContiguousGpuTensor(values_out, "values_out");
    requireSameGpuPlacement(keys_in, values_in, "keys_in", "values_in");
    requireSameGpuPlacement(keys_in, values_out, "keys_in", "values_out");
    requireStorageForNumItems(values_in, "values_in", total_items);
    requireStorageForNumItems(values_out, "values_out", output_items);
    if (values_in.getDataType() != values_out.getDataType()) {
        throw std::invalid_argument("CUB segmented top-k value input/output dtypes must match.");
    }
    if (!isCubTopKValueDTypeSupported(values_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented top-k index value dtype " + dtypeName(values_in.getDataType()) + ".");
    }
}

void validateSegmentOffsets(const Tensor& reference,
                            const Tensor& segment_offsets,
                            uint64_t num_items,
                            uint64_t num_segments,
                            const char* reference_name) {
    requireDenseContiguousGpuTensor(segment_offsets, "segment_offsets");
    requireSameGpuPlacement(reference, segment_offsets, reference_name, "segment_offsets");
    if (num_segments == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("num_segments is too large for a contiguous segment-offset tensor.");
    }
    requireStorageForNumItems(segment_offsets, "segment_offsets", num_segments + 1);
    if (!isCubSegmentOffsetDTypeSupported(segment_offsets.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segment-offset dtype " + dtypeName(segment_offsets.getDataType()) + ".");
    }
    static_cast<void>(checkedCubInt64Count(num_items, "num_items"));
    static_cast<void>(checkedCubInt64Count(num_segments, "num_segments"));
}

void validateSegmentOffsets(const Tensor& keys_in, const Tensor& segment_offsets, uint64_t num_items, uint64_t num_segments) {
    validateSegmentOffsets(keys_in, segment_offsets, num_items, num_segments, "keys_in");
}

void validateSegmentedSortKeys(const Tensor& keys_in,
                               const Tensor& keys_out,
                               const Tensor& segment_offsets,
                               uint64_t num_items,
                               uint64_t num_segments,
                               int begin_bit,
                               int end_bit) {
    validateSortKeys(keys_in, keys_out, num_items, begin_bit, end_bit);
    validateSegmentOffsets(keys_in, segment_offsets, num_items, num_segments);
}

void validateSegmentedSortPairs(const Tensor& keys_in,
                                const Tensor& keys_out,
                                const Tensor& values_in,
                                const Tensor& values_out,
                                const Tensor& segment_offsets,
                                uint64_t num_items,
                                uint64_t num_segments,
                                int begin_bit,
                                int end_bit) {
    validateSortPairs(keys_in, keys_out, values_in, values_out, num_items, begin_bit, end_bit);
    validateSegmentOffsets(keys_in, segment_offsets, num_items, num_segments);
}

void validateRle(const Tensor& input, const Tensor& unique_out, const Tensor& counts_out, const Tensor& num_runs_out, uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(unique_out, "unique_out");
    requireDenseContiguousGpuTensor(counts_out, "counts_out");
    requireDenseContiguousGpuTensor(num_runs_out, "num_runs_out");
    requireSameGpuPlacement(input, unique_out, "input", "unique_out");
    requireSameGpuPlacement(input, counts_out, "input", "counts_out");
    requireSameGpuPlacement(input, num_runs_out, "input", "num_runs_out");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(unique_out, "unique_out", num_items);
    requireStorageForNumItems(counts_out, "counts_out", num_items);
    requireStorageForNumItems(num_runs_out, "num_runs_out", 1);
    if (input.getDataType() != unique_out.getDataType()) {
        throw std::invalid_argument("CUB RLE input and unique_out dtypes must match.");
    }
    if (!isCubRunLengthEncodeDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB RLE dtype " + dtypeName(input.getDataType()) + ".");
    }
    if (counts_out.getDataType() != DataType::UINT32 || num_runs_out.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB RLE counts_out and num_runs_out must have dtype uint32.");
    }
}

namespace {

void validateDeviceReduceCommon(const Tensor& input, const Tensor& output, uint64_t num_items, const char* op_name) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", 1);
    if (num_items == 0) {
        throw std::invalid_argument(std::string("CUB device ") + op_name + " requires num_items to be nonzero.");
    }
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument(std::string("CUB device ") + op_name + " input/output dtypes must match.");
    }
    static_cast<void>(checkedCubNumItems(num_items));
}

}  // namespace

void validateDeviceReduceSum(const Tensor& input, const Tensor& output, uint64_t num_items) {
    validateDeviceReduceCommon(input, output, num_items, "reduce-sum");
    if (!isCubReduceSumDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB device reduce-sum dtype " + dtypeName(input.getDataType()) + ".");
    }
}

void validateDeviceReduceMax(const Tensor& input, const Tensor& output, uint64_t num_items) {
    validateDeviceReduceCommon(input, output, num_items, "reduce-max");
    if (!isCubReduceMaxDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB device reduce-max dtype " + dtypeName(input.getDataType()) + ".");
    }
}

void validateDeviceReduceMin(const Tensor& input, const Tensor& output, uint64_t num_items) {
    validateDeviceReduceCommon(input, output, num_items, "reduce-min");
    if (!isCubReduceMinDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB device reduce-min dtype " + dtypeName(input.getDataType()) + ".");
    }
}

void validateExclusiveSum(const Tensor& input, const Tensor& output, uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", num_items);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument("CUB exclusive-sum input/output dtypes must match.");
    }
    if (!isCubExclusiveSumDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB exclusive-sum dtype " + dtypeName(input.getDataType()) + ".");
    }
}

void validateSegmentedExclusiveSum(const Tensor& input,
                                   const Tensor& output,
                                   const Tensor& segment_offsets,
                                   uint64_t num_items,
                                   uint64_t num_segments) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", num_items);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument("CUB segmented exclusive-sum input/output dtypes must match.");
    }
    if (!isCubSegmentedExclusiveSumDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented exclusive-sum dtype " + dtypeName(input.getDataType()) + ".");
    }
    validateSegmentOffsets(input, segment_offsets, num_items, num_segments, "input");
}

namespace {

void validateSegmentedReduceCommon(const Tensor& input,
                                   const Tensor& output,
                                   const Tensor& segment_offsets,
                                   uint64_t num_items,
                                   uint64_t num_segments,
                                   const char* op_name) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", num_segments);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument(std::string("CUB segmented ") + op_name + " input/output dtypes must match.");
    }
    validateSegmentOffsets(input, segment_offsets, num_items, num_segments, "input");
}

}  // namespace

void validateSegmentedReduceSum(const Tensor& input,
                                const Tensor& output,
                                const Tensor& segment_offsets,
                                uint64_t num_items,
                                uint64_t num_segments) {
    validateSegmentedReduceCommon(input, output, segment_offsets, num_items, num_segments, "reduce-sum");
    if (!isCubSegmentedReduceSumDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented reduce-sum dtype " + dtypeName(input.getDataType()) + ".");
    }
}

void validateSegmentedReduceMax(const Tensor& input,
                                const Tensor& output,
                                const Tensor& segment_offsets,
                                uint64_t num_items,
                                uint64_t num_segments) {
    validateSegmentedReduceCommon(input, output, segment_offsets, num_items, num_segments, "reduce-max");
    if (!isCubSegmentedReduceMaxDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented reduce-max dtype " + dtypeName(input.getDataType()) + ".");
    }
}

}  // namespace ThorImplementation::CubDevicePrimitiveSupport

namespace ThorImplementation {

bool isCubRadixSortKeyDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FP64:
            return true;
#endif
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubRadixSortValueDTypeSupported(DataType dtype) {
    // SortPairs is Thor's internal argsort/segmented-argsort backend: the
    // carried payload is an index vector, not an arbitrary user value tensor.
    // Arbitrary value reordering belongs in take_along_axis after argsort.
    // Keeping pair values to unsigned index dtypes avoids a key-dtype x
    // arbitrary-value-dtype template-instantiation explosion in CUB.
    if (dtype == DataType::UINT32) {
        return true;
    }
#if THOR_CUB_ENABLE_64BIT_TYPES
    if (dtype == DataType::UINT64) {
        return true;
    }
#endif
    return false;
}

bool isCubTopKKeyDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FP64:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubTopKValueDTypeSupported(DataType dtype) {
    return isCubRadixSortValueDTypeSupported(dtype);
}

bool isCubSelectValueDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FP64:
            return true;
#endif
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubSelectFlagDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
            return true;
        default:
            return false;
    }
}

bool isCubFindKeyDTypeSupported(DataType dtype) { return isCubTopKKeyDTypeSupported(dtype); }

bool isCubFindFlagDTypeSupported(DataType dtype) { return isCubSelectFlagDTypeSupported(dtype); }

bool isCubRunLengthEncodeDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FP64:
            return true;
#endif
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubExclusiveSumDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::FP64:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubScanDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubReduceSumDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubReduceMaxDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubReduceMinDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubSegmentedExclusiveSumDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubSegmentedReduceSumDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubSegmentedReduceMaxDTypeSupported(DataType dtype) { return isCubExclusiveSumDTypeSupported(dtype); }

bool isCubSegmentOffsetDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return true;
#endif
        default:
            return false;
    }
}

CubTemporaryStoragePlan cubTemporaryStoragePlan(const TensorPlacement& placement, size_t bytes) {
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("CUB temporary storage must be planned on GPU placement.");
    }
    CubTemporaryStoragePlan plan;
    plan.placement = placement;
    plan.bytes = std::max<size_t>(bytes, 1);
    return plan;
}

CubTemporaryStoragePlan cubMaxTemporaryStoragePlan(std::initializer_list<CubTemporaryStoragePlan> plans) {
    if (plans.size() == 0) {
        throw std::invalid_argument("At least one CUB temporary-storage plan is required.");
    }

    auto it = plans.begin();
    CubTemporaryStoragePlan combined = cubTemporaryStoragePlan(it->placement, it->bytes);
    ++it;
    for (; it != plans.end(); ++it) {
        if (it->placement != combined.placement) {
            throw std::invalid_argument("All CUB temporary-storage plans must use the same GPU placement.");
        }
        combined.bytes = std::max(combined.bytes, std::max<size_t>(it->bytes, 1));
    }
    return combined;
}

Tensor allocateCubTemporaryStorage(const TensorPlacement& placement, size_t bytes) {
    const CubTemporaryStoragePlan plan = cubTemporaryStoragePlan(placement, bytes);
    return allocateCubTemporaryStorage(plan);
}

Tensor allocateCubTemporaryStorage(const CubTemporaryStoragePlan& plan) {
    if (plan.placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("CUB temporary storage must be allocated on GPU.");
    }
    return Tensor(plan.placement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(std::max<size_t>(plan.bytes, 1))}));
}

}  // namespace ThorImplementation
