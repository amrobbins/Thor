// DEPRECATED
// #include "Utilities/Common/ScopedGpu.h"
// #include "Utilities/Common/Stream.h"
//
// #include <cuda.h>
// #include <cuda_fp16.h>
//
// #include <assert.h>
// #include <cstdint>
// #include <type_traits>
//
// template <typename T>
// void launchAdamStep(T *weightUpdate_d,
//                     T *gradient_d,
//                     float *m_d,
//                     float *v_d,
//                     float t,
//                     float alpha,
//                     float beta1,
//                     float beta2,
//                     float epsilon,
//                     uint32_t length,
//                     float inverseBatchSizeTimesInverseLossScale,
//                     Stream stream);
