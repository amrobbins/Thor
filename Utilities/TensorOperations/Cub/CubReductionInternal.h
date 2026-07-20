#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstddef>

namespace ThorImplementation::CubReductionInternal {

size_t querySumReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchSumReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryProductReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchProductReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryMeanReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchMeanReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryL1NormReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchL1NormReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryL2NormReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchL2NormReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryMinReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchMinReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryMaxReductionBytes(DataType input_dtype,
                              const void* input,
                              uint64_t input_elements,
                              DataType output_dtype,
                              void* output,
                              const CubReductionGeometry& geometry,
                              float output_scale,
                              const Stream& stream);
void launchMaxReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        float output_scale,
                        Stream& stream);

size_t queryOffsetSegmentedReductionBytes(CubReductionOp op,
                                          const Tensor& input,
                                          Tensor& output,
                                          const Tensor& segment_offsets,
                                          uint64_t num_items,
                                          uint64_t num_segments,
                                          const Stream& stream);
void launchOffsetSegmentedReduction(CubReductionOp op,
                                    const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    Stream& stream);

size_t queryArgMinReductionBytes(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
void launchArgMinReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           Stream& stream);

size_t queryArgMaxReductionBytes(const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
void launchArgMaxReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor* value_output,
                           Tensor* index_output,
                           const CubReductionGeometry& geometry,
                           Stream& stream);

}  // namespace ThorImplementation::CubReductionInternal
