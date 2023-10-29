/*
 *
 * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html
 * void cblas_hgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT
 * n, const MKL_INT k, const MKL_F16 alpha, const MKL_F16 *a, const MKL_INT lda, const MKL_F16 *b, const MKL_INT ldb, const MKL_F16 beta,
 * MKL_F16 *c, const MKL_INT ldc);
 *
 * void cblas_sgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT
 * n, const MKL_INT k, const float alpha, const float *a, const MKL_INT lda, const float *b, const MKL_INT ldb, const float beta, float *c,
 * const MKL_INT ldc);
 *
 * Layout
 * Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
 * We want row major since this is c++
 *
 *
 * Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
 * https://oneapi-src.github.io/oneDNN/v0/index.html
 * This will be the CPU version of cudnn.
 * This has its own streams... There is a whole parallel implementation that needs to happen for cpu processing.
 * Thor's Stream needs to be updated to use MKL streams for CPU compute and to interface/schedule between CPU and GPU streams.
 * Or possibly could just use Stream::enqueueHostFunction(...), however this would preclude async host functions from interacting with Cuda,
 * maybe that's ok since a layer's implementation should be either CPU or GPU, but what if there is a use case for a hybrid implementation?
 * On further inspection, MKL streams don't provide much utility and don't seem appropriate for this framework, so cpu will
 * rely on Stream::enqueueHostFunction(...), so a cpu kernel cannot interact with cuda directly in the enqueued kernel, however
 * memory can be written to by the cpu kernel which then can be fed into cuda by a different kernel enqueued to the stream that is aware
 * of the cpu output tensor.
 */