#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <map>
#include <utility>
#include <vector>

using std::map;
using std::pair;
using std::vector;

void launchPad(half *dest_d,
               half *source_d,
               unsigned long numDestElements,
               unsigned int numDimensions,
               unsigned long stridePerPaddedDimension_d[],
               unsigned long stridePerUnpaddedDimension_d[],
               unsigned int padBefore_d[],
               unsigned int padAfter_d[],
               Stream stream);
