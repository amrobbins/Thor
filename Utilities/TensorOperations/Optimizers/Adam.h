#include "Utilities/Common/Stream.h"

#include <cuda.h>
#include <cuda_fp16.h>

#include <assert.h>
#include <cstdint>
#include <type_traits>

template <typename T>
void launchAdamStep(
    T *weightUpdate_d, T *gradient_d, T *m_d, T *v_d, float t, T alpha, T beta1, T beta2, T epsilon, uint32_t length, Stream stream);
