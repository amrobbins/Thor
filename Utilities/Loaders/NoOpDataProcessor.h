#pragma once

/*
#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"
*/

#include <memory.h>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"

class NoOpDataProcessor : public DataProcessor {
   public:
    NoOpDataProcessor(uint64_t numOutputTensorBytes, ThorImplementation::TensorDescriptor::DataType dataType);
    virtual ~NoOpDataProcessor() {}

    virtual uint64_t outputTensorSizeInBytes();
    virtual ThorImplementation::TensorDescriptor::DataType getDataType();

    virtual DataElement operator()(DataElement &input);

   private:
    uint64_t numOutputTensorBytes;
    ThorImplementation::TensorDescriptor::DataType dataType;
};
