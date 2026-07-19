#pragma once

#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cstddef>

namespace ThorImplementation::CubReductionInternal {

size_t querySumReductionBytes(const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              const Stream& stream);
void launchSumReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        Stream& stream);

size_t queryProductReductionBytes(const Tensor& input,
                                  Tensor& output,
                                  const CubReductionGeometry& geometry,
                                  const Stream& stream);
void launchProductReduction(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& input,
                            Tensor& output,
                            const CubReductionGeometry& geometry,
                            Stream& stream);

size_t queryMeanReductionBytes(const Tensor& input,
                               Tensor& output,
                               const CubReductionGeometry& geometry,
                               const Stream& stream);
void launchMeanReduction(const Tensor& temp_storage,
                         size_t temp_storage_bytes,
                         const Tensor& input,
                         Tensor& output,
                         const CubReductionGeometry& geometry,
                         Stream& stream);

size_t queryL1NormReductionBytes(const Tensor& input,
                                 Tensor& output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
void launchL1NormReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           const CubReductionGeometry& geometry,
                           Stream& stream);

size_t queryL2NormReductionBytes(const Tensor& input,
                                 Tensor& output,
                                 const CubReductionGeometry& geometry,
                                 const Stream& stream);
void launchL2NormReduction(const Tensor& temp_storage,
                           size_t temp_storage_bytes,
                           const Tensor& input,
                           Tensor& output,
                           const CubReductionGeometry& geometry,
                           Stream& stream);

size_t queryMinReductionBytes(const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              const Stream& stream);
void launchMinReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        Stream& stream);

size_t queryMaxReductionBytes(const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              const Stream& stream);
void launchMaxReduction(const Tensor& temp_storage,
                        size_t temp_storage_bytes,
                        const Tensor& input,
                        Tensor& output,
                        const CubReductionGeometry& geometry,
                        Stream& stream);

}  // namespace ThorImplementation::CubReductionInternal
