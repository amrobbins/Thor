#pragma once

void swap(int &a, int &b) {
    int c = a;
    a = b;
    b = c;
}

void verifyOperationIsLegal(int rowsA, int colsA, int rowsB, int colsB, int ldA, int ldB, int ldC, int transposeA, int transposeB) {
    if (transposeA) {
        swap(rowsA, colsA);
    }
    if (transposeB) {
        swap(rowsB, colsB);
    }

    assert(colsA == rowsB);
    assert(ldC >= colsB);
}

void transpose(float *A, float *A_t, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[j * rows + i] = A_t[i * ld + j];
        }
    }
}

void transposeHalf(half *A, half *A_t, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[j * rows + i] = A_t[i * ld + j];
        }
    }
}

// computes result = AB, where A is an mxk matrix and B is an kxn matrix. This makes result a mxn matrix.
void matrixMultiplyCpu(float *A,
                       float *B,
                       float *C,
                       int rowsA,
                       int colsA,
                       int rowsB,
                       int colsB,
                       int lda,
                       int ldb,
                       int ldc,
                       bool transposeA,
                       bool transposeB,
                       bool accumulate) {
    omp_set_num_threads(10);

    float *A_t = nullptr;
    if (transposeA) {
        A_t = A;
        A = new float[rowsA * colsA];

        transpose(A, A_t, rowsA, colsA, lda);
        swap(rowsA, colsA);
        lda = colsA;
    }

    float *B_t = nullptr;
    if (transposeB) {
        B_t = B;
        B = new float[rowsB * colsB];

        transpose(B, B_t, rowsB, colsB, ldb);
        swap(rowsB, colsB);
        ldb = colsB;
    }

    verifyOperationIsLegal(rowsA, colsA, rowsB, colsB, lda, ldb, ldc, false, false);

#pragma omp parallel for schedule(static, 3)
    for (int ra = 0; ra < rowsA; ra++) {
        for (int cb = 0; cb < colsB; cb++) {
            float accum = 0.0;
            for (int carb = 0; carb < colsA; carb++)
                accum += A[ra * lda + carb] * B[carb * ldb + cb];

            if (accumulate)
                C[ra * ldc + cb] += accum;
            else
                C[ra * ldc + cb] = accum;
        }
    }

    if (transposeA)
        delete A;
    if (transposeB)
        delete B;
}

void matrixMultiplyCpuHalf(half *A,
                           half *B,
                           half *C,
                           int rowsA,
                           int colsA,
                           int rowsB,
                           int colsB,
                           int lda,
                           int ldb,
                           int ldc,
                           bool transposeA,
                           bool transposeB,
                           bool accumulate) {
    omp_set_num_threads(10);

    half *A_t = nullptr;
    if (transposeA) {
        A_t = A;
        A = new half[rowsA * colsA];

        transposeHalf(A, A_t, rowsA, colsA, lda);
        swap(rowsA, colsA);
        lda = colsA;
    }

    half *B_t = nullptr;
    if (transposeB) {
        B_t = B;
        B = new half[rowsB * colsB];

        transposeHalf(B, B_t, rowsB, colsB, ldb);
        swap(rowsB, colsB);
        ldb = colsB;
    }

    assert(colsA == rowsB);

#pragma omp parallel for schedule(static, 3)
    for (int ra = 0; ra < rowsA; ra++) {
        for (int cb = 0; cb < colsB; cb++) {
            float accum = 0.0;
            for (int carb = 0; carb < colsA; carb++)
                accum += A[ra * lda + carb] * B[carb * ldb + cb];

            if (accumulate)
                C[ra * ldc + cb] = (float)(C[ra * ldc + cb]) + accum;
            else
                C[ra * ldc + cb] = accum;
        }
    }

    if (transposeA)
        delete A;
    if (transposeB)
        delete B;
}

void printMatrix(float *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%5.2f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrixWide(float *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrix(half *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%5.2f ", (float)matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrixWide(half *matrix, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%7.2f ", (float)matrix[i * ld + j]);
        }
        printf("\n");
    }
}

void printMatrices(float *matrixCpu, float *matrixGpu, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i) {
        printf("CPU: ");
        for (int j = 0; j < cols; ++j) {
            printf("%5.2f:%-5.2f", matrixCpu[i * cols + j], matrixGpu[i * cols + j]);
        }
        printf(" :GPU\n");
    }
}
