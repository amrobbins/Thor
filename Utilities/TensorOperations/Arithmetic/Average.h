#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

// source_d_pd[] is a device memory array of pointers to device memory
// i.e. a variable on the host that holds the address of a device memory array of pointers to each of the device memory arrays of source
// data. dest_d is a device memory array where the result will be stored.
void launchAverage(half *dest_d, half *source_d_pd[], int numInstances, int numElements, Stream stream);
