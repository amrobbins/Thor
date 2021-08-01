#pragma once

#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

void launchSumScaleHalfAll(half result_d[], half noScaleSource[], half scaleSource[], half scale, uint64_t numElements, Stream stream);

void launchSumScaleHalfSourceDestScaleSource(
    half result_d[], half noScaleSource[], half scaleSource[], float scale, uint64_t numElements, Stream stream);

void launchSumScaleHalfSourceDest(
    half result_d[], half noScaleSource[], float scaleSource[], float scale, uint64_t numElements, Stream stream);

template <typename DEST_TYPE, typename NO_SCALE_SOURCE_TYPE, typename SCALE_SOURCE_TYPE, typename SCALE_TYPE>
void launchSumScale(DEST_TYPE result_d[],
                    NO_SCALE_SOURCE_TYPE noScaleSource[],
                    SCALE_SOURCE_TYPE scaleSource[],
                    SCALE_TYPE scale,
                    uint64_t numElements,
                    Stream stream);
