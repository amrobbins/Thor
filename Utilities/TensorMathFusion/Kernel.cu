/* Expected shape

extern "C" __global__
void fused_kernel(const float* in0,
                  const float* in1,
                  const float* in2,
                  float* out,
                  unsigned long long numel) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel)
        return;

    float v0 = in0[idx];
    float v1 = in1[idx];
    float v2 = in2[idx];

    float t3 = v0 / v1;
    float t4 = t3 * v2;
    float t5 = 1.5f;
    float t6 = t4 + t5;

    out[idx] = t6;
}
*/
