#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

TEST(Average, AveragesCorrectly) {
    srand(time(NULL));

    half source[10][4096];
    half dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half **source_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source_d, 10 * sizeof(half *));
    assert(cudaStatus == cudaSuccess);
    half *sourceArray[10];
    for (int i = 0; i < 10; ++i) {
        cudaStatus = cudaMalloc(&(sourceArray[i]), 4096 * sizeof(half));
        assert(cudaStatus == cudaSuccess);
    }
    cudaStatus = cudaMemcpyAsync(source_d, sourceArray, 10 * sizeof(half *), cudaMemcpyHostToDevice, stream.getStream());
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numInstances = (rand() % 10) + 1;
        int numElements = (rand() % 4096) + 1;

        for (int i = 0; i < numInstances; ++i) {
            for (int j = 0; j < numElements; ++j) {
                source[i][j] = ((rand() % 100) / 10.0f) - 5.0f;
            }
            cudaStatus = cudaMemcpyAsync(sourceArray[i], source[i], numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
            assert(cudaStatus == cudaSuccess);
        }

        launchAverage(dest_d, source_d, numInstances, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numInstances; ++i) {
            for (int j = 0; j < numElements; ++j) {
                if (i == 0)
                    dest_cpu[j] = (float)source[i][j];
                else
                    dest_cpu[j] = dest_cpu[j] + (float)source[i][j];
                if (i == numInstances - 1)
                    dest_cpu[j] = dest_cpu[j] / numInstances;
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)(half)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f", numElements, i, (float)dest[i], (float)(half)dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    for (int i = 0; i < 10; ++i) {
        cudaStatus = cudaFree(sourceArray[i]);
        assert(cudaStatus == cudaSuccess);
    }
    cudaStatus = cudaFree(source_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(Sum, SumsCorrectly) {
    srand(time(NULL));

    half source[10][4096];
    half dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half **source_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source_d, 10 * sizeof(half *));
    assert(cudaStatus == cudaSuccess);
    half *sourceArray[10];
    for (int i = 0; i < 10; ++i) {
        cudaStatus = cudaMalloc(&(sourceArray[i]), 4096 * sizeof(half));
        assert(cudaStatus == cudaSuccess);
    }
    cudaStatus = cudaMemcpyAsync(source_d, sourceArray, 10 * sizeof(half *), cudaMemcpyHostToDevice, stream.getStream());
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numInstances = (rand() % 10) + 1;
        int numElements = (rand() % 4096) + 1;

        for (int i = 0; i < numInstances; ++i) {
            for (int j = 0; j < numElements; ++j) {
                source[i][j] = ((rand() % 100) / 10.0f) - 5.0f;
            }
            cudaStatus = cudaMemcpyAsync(sourceArray[i], source[i], numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
            assert(cudaStatus == cudaSuccess);
        }

        launchSum(dest_d, source_d, numInstances, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numInstances; ++i) {
            for (int j = 0; j < numElements; ++j) {
                if (i == 0)
                    dest_cpu[j] = (float)source[i][j];
                else
                    dest_cpu[j] = dest_cpu[j] + (float)source[i][j];
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)(half)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f", numElements, i, (float)dest[i], (float)(half)dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    for (int i = 0; i < 10; ++i) {
        cudaStatus = cudaFree(sourceArray[i]);
        assert(cudaStatus == cudaSuccess);
    }
    cudaStatus = cudaFree(source_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(SumScaleTemplate, ComputesCorrectAnswer) {
    srand(time(NULL));

    float source0[4096];
    float source1[4096];
    float dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    float *source0_d;
    float *source1_d;
    float *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source0_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source1_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numElements = (rand() % 4096) + 1;
        float scale = ((rand() % 100) / 10.0f) - 5.0f;
        for (int i = 0; i < numElements; ++i) {
            source0[i] = ((rand() % 100) / 10.0f) - 5.0f;
            source1[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        cudaStatus = cudaMemcpyAsync(source0_d, source0, numElements * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(source1_d, source1, numElements * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchSumScale(dest_d, source0_d, source1_d, scale, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i)
            dest_cpu[i] = ((float)source0[i] + scale * (float)source1[i]);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f", numElements, i, (float)dest[i], (float)dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(source0_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(source1_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(SumScaleHalfSourceDest, ComputesCorrectAnswer) {
    srand(time(NULL));

    half source0[4096];
    float source1[4096];
    half dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *source0_d;
    float *source1_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source0_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source1_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numElements = (rand() % 4096) + 1;
        float scale = ((rand() % 100) / 10.0f) - 5.0f;
        for (int i = 0; i < numElements; ++i) {
            source0[i] = ((rand() % 100) / 10.0f) - 5.0f;
            source1[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        cudaStatus = cudaMemcpyAsync(source0_d, source0, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(source1_d, source1, numElements * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchSumScaleHalfSourceDest(dest_d, source0_d, source1_d, scale, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i)
            dest_cpu[i] = ((float)source0[i] + scale * source1[i]);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)(half)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f\n", numElements, i, (float)dest[i], (float)(half)dest_cpu[i]);
            }
            EXPECT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(source0_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(source1_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(SumScaleHalfSourceDestScaleSource, ComputesCorrectAnswer) {
    srand(time(NULL));

    half source0[4096];
    half source1[4096];
    half dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *source0_d;
    half *source1_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source0_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source1_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numElements = (rand() % 4096) + 1;
        float scale = ((rand() % 100) / 10.0f) - 5.0f;
        for (int i = 0; i < numElements; ++i) {
            source0[i] = ((rand() % 100) / 10.0f) - 5.0f;
            source1[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        cudaStatus = cudaMemcpyAsync(source0_d, source0, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(source1_d, source1, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchSumScaleHalfSourceDestScaleSource(dest_d, source0_d, source1_d, scale, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i)
            dest_cpu[i] = ((float)source0[i] + scale * (float)source1[i]);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)(half)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f", numElements, i, (float)dest[i], (float)(half)dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(source0_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(source1_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(SumScaleHalfAll, ComputesCorrectAnswer) {
    srand(time(NULL));

    half source0[4096];
    half source1[4096];
    half dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *source0_d;
    half *source1_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source0_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source1_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numElements = (rand() % 4096) + 1;
        half scale = ((rand() % 100) / 10.0f) - 5.0f;
        for (int i = 0; i < numElements; ++i) {
            source0[i] = ((rand() % 100) / 10.0f) - 5.0f;
            source1[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        cudaStatus = cudaMemcpyAsync(source0_d, source0, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(source1_d, source1, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchSumScaleHalfAll(dest_d, source0_d, source1_d, scale, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i)
            dest_cpu[i] = ((float)source0[i] + (float)scale * (float)source1[i]);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)(half)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f", numElements, i, (float)dest[i], (float)(half)dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(source0_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(source1_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(Exponentiation, ComputesCorrectAnswer) {
    srand(time(NULL));

    half exponent[4096];
    float dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *exponent_d;
    float *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&exponent_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numElements = (rand() % 4096) + 1;
        for (int i = 0; i < numElements; ++i) {
            exponent[i] = ((rand() % 100) / 20.0f) - 2.5f;
        }
        cudaStatus = cudaMemcpyAsync(exponent_d, exponent, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchExponentiation(exponent_d, dest_d, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i)
            dest_cpu[i] = exp((float)exponent[i]);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs(dest[i] - dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f\n", numElements, i, dest[i], dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(exponent_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(SumManyToOne, ComputesCorrectAnswer) {
    srand(time(NULL));

    half summand[4096];
    float dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *summand_d;
    float *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&summand_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int batchSize = (rand() % 8) + 1;
        int numElementsPerBatch = (rand() % 512) + 1;
        bool invert = rand() % 2 ? true : false;
        bool accumulate = rand() % 2 ? true : false;
        for (int i = 0; i < numElementsPerBatch * batchSize; ++i) {
            summand[i] = ((rand() % 100) / 20.0f) - 2.5f;
        }

        for (int b = 0; b < batchSize; ++b)
            dest_cpu[b] = ((rand() % 100) / 20.0f) - 2.5f;
        cudaStatus = cudaMemcpyAsync(dest_d, dest_cpu, batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        stream.synchronize();

        for (int b = 0; b < batchSize; ++b) {
            if (!accumulate)
                dest_cpu[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i)
                dest_cpu[b] += (float)summand[b * numElementsPerBatch + i];
            if (invert && abs(dest_cpu[b]) < 0.1f) {
                summand[b * numElementsPerBatch] = summand[b * numElementsPerBatch] + (half)1.0f;
                dest_cpu[b] += 1.0f;
            }
            if (invert)
                dest_cpu[b] = 1.0f / dest_cpu[b];
        }

        cudaStatus =
            cudaMemcpyAsync(summand_d, summand, numElementsPerBatch * batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchSumManyToOne(summand_d, dest_d, numElementsPerBatch, batchSize, invert, accumulate, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < batchSize; ++i) {
            float diff = abs(dest[i] - dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElementsPerBatch %d element %d %f %f\n", numElementsPerBatch, i, dest[i], dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(summand_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(MultiplyByScalar, ComputesCorrectAnswer) {
    srand(time(NULL));

    half source[4096];
    half batchedScalars[8];
    float dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *source_d;
    half *batchedScalars_d;
    float *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&batchedScalars_d, 8 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int batchSize = (rand() % 8) + 1;
        int numElementsPerBatch = (rand() % 512) + 1;
        for (int b = 0; b < batchSize; ++b) {
            batchedScalars[b] = ((rand() % 100) / 10.0f) - 5.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                source[i + b * numElementsPerBatch] = ((rand() % 100) / 10.0f) - 5.0f;
            }
        }

        cudaStatus =
            cudaMemcpyAsync(source_d, source, numElementsPerBatch * batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpyAsync(batchedScalars_d, batchedScalars, batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchMultiplyByScalar(source_d, batchedScalars_d, dest_d, numElementsPerBatch, batchSize, stream);

        cudaStatus =
            cudaMemcpyAsync(dest, dest_d, numElementsPerBatch * batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i)
                dest_cpu[b * numElementsPerBatch + i] = (float)source[b * numElementsPerBatch + i] * (float)batchedScalars[b];
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < batchSize; ++i) {
            float diff = abs(dest[i] - dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElementsPerBatch %d element %d %f %f\n", numElementsPerBatch, i, dest[i], dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(source_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(batchedScalars_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(ElementWiseSubtract, ComputesCorrectAnswer) {
    srand(time(NULL));

    half source0[4096];
    half source1[4096];
    float dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half *source0_d;
    half *source1_d;
    float *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source0_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source1_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numElements = (rand() % 4096) + 1;
        for (int i = 0; i < numElements; ++i) {
            source0[i] = ((rand() % 100) / 20.0f) - 2.5f;
            source1[i] = ((rand() % 100) / 20.0f) - 2.5f;
        }
        cudaStatus = cudaMemcpyAsync(source0_d, source0, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(source1_d, source1, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchElementwiseSubtract(source0_d, source1_d, dest_d, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i)
            dest_cpu[i] = (float)source0[i] - (float)source1[i];

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.1;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs(dest[i] - dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f\n", numElements, i, dest[i], dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(source0_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}
