#pragma once

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Loaders/ImageLoader.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"

#include <cuda_fp16.h>

#include <memory>
#include <mutex>

class ImageProcessor : public DataProcessor {
   public:
    ImageProcessor(double minAspectRatio,
                   double maxAspectRatio,
                   uint32_t outputImageRows,
                   uint32_t outputImageColumns,
                   uint32_t bytesPerPixelChannel = 2,
                   bool centerAndScale = true,
                   bool display = false);

    ImageProcessor(double minAspectRatio,
                   double maxAspectRatio,
                   uint32_t outputImageRows,
                   uint32_t outputImageColumns,
                   bool (*customProcessor)(uint8_t *regularlyProcessedImage),
                   bool display = false);

    ImageProcessor(double minAspectRatio,
                   double maxAspectRatio,
                   uint32_t outputImageRows,
                   uint32_t outputImageColumns,
                   bool (*customProcessor)(half *regularlyProcessedImage),
                   bool centerAndScale = true,
                   bool display = false);

    virtual ~ImageProcessor() {}

    virtual uint64_t outputTensorSizeInBytes();
    virtual uint64_t outputTensorSizeInPixels();
    virtual ThorImplementation::TensorDescriptor::DataType getDataType();

    virtual DataElement operator()(DataElement &input);

   private:
    double minAspectRatio;
    double maxAspectRatio;
    double cropCenterToAspectRatio;
    uint32_t outputImageRows;
    uint32_t outputImageColumns;
    bool (*customProcessorUint8)(uint8_t *regularlyProcessedImage);
    bool (*customProcessorHalf)(half *regularlyProcessedImage);
    bool display;
    uint32_t bytesPerPixelChannel;
    bool centerAndScale;

    std::mutex mutex;
};
