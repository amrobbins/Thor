#include "Thor.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ThorImplementation;

TEST(BatchReduce, reduce) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    Stream stream(0);

    for (uint32_t i = 0; i < 10; ++i) {
        uint32_t dim0 = (rand() % 500) + 1;
        uint32_t dim1 = (rand() % 100) + 1;
        Tensor sourceT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dim0, dim1}));
        Tensor destT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dim1}));
        Tensor brdestT(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dim1}));
        Tensor brdestTCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {dim1}));
        float *source = (float *)sourceT.getMemPtr();
        float *dest = (float *)destT.getMemPtr();

        for (uint32_t row = 0; row < dim0; ++row) {
            for (uint32_t col = 0; col < dim1; ++col) {
                float val = (float)(rand() % 10) / ((float)(rand() % 10) + 1);
                source[row * dim1 + col] = val;

                if (row == 0)
                    dest[col] = 0.0f;
                dest[col] += val;
            }
        }

        brdestT.copyFromAsync(destT, stream);
        BatchReduce batchReduce(dim0,
                                dim0,
                                dim1,
                                ThorImplementation::TensorDescriptor::DataType::FP32,
                                ThorImplementation::TensorDescriptor::DataType::FP32,
                                ThorImplementation::TensorDescriptor::DataType::FP32,
                                stream);
        batchReduce.reduce(source, brdestT.getMemPtr());
        brdestTCpu.copyFromAsync(brdestT, stream);
        stream.synchronize();

        float *brdest = (float *)brdestTCpu.getMemPtr();
        const float thresh = 0.001;
        for (uint32_t col = 0; col < dim1; ++col) {
            ASSERT_LT(abs(dest[col] / (float)dim0 - brdest[col]), thresh);
            printf("%f %f\n", dest[col] / (float)dim0, brdest[col]);
        }
    }
}

TEST(BatchReduce, getStream) {
    Stream stream(0);
    Stream stream2(0);
    BatchReduce batchReduce(128,
                            256,
                            50,
                            ThorImplementation::TensorDescriptor::DataType::FP32,
                            ThorImplementation::TensorDescriptor::DataType::FP16,
                            ThorImplementation::TensorDescriptor::DataType::FP32,
                            stream);

    assert(stream == batchReduce.getStream());
    assert(stream2 != batchReduce.getStream());
}