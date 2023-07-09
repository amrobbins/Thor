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
 */