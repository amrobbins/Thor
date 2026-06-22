#pragma once

#include "Utilities/Common/Stream.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

template <typename INDEX_TYPE>
void launchMap(void *dest_d,
               void *source_d,
               INDEX_TYPE *mapping_d,
               INDEX_TYPE numDestElements,
               ThorImplementation::DataType dataType,
               Stream stream);

template <typename INDEX_TYPE>
void launchMapNInto1(unsigned int N,
                     void *dest_d,
                     void *source_d,
                     INDEX_TYPE *mapping_d,
                     INDEX_TYPE numMapElements,
                     ThorImplementation::DataType dataType,
                     Stream stream);
