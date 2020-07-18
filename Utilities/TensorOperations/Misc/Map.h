#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

template <typename INDEX_TYPE>
void launchMap(half *dest_d, half *source_d, INDEX_TYPE *mapping_d, INDEX_TYPE numDestElements, Stream stream);

template <typename INDEX_TYPE>
void launchMapNInto1(unsigned int N, half *dest_d, half *source_d, INDEX_TYPE *mapping_d, INDEX_TYPE numMapElements, Stream stream);
