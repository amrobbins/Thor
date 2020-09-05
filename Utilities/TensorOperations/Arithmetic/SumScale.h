#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchSumScale(half result_d[], half noScaleSource[], half scaleSource[], float scale, int numElements, Stream stream);
void launchSumScale(half result_d[], half noScaleSource[], float scaleSource[], float scale, int numElements, Stream stream);
