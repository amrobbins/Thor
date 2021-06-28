#pragma once

#include "Utilities/Loaders/ImageLoader.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"

#include <memory>
#include <mutex>

class ImageProcessor : public DataProcessor {
   public:
    ImageProcessor(double minAspectRatio,
                   double maxAspectRatio,
                   uint32_t outputImageRows,
                   uint32_t outputImageColumns,
                   bool (*customProcessor)(Magick::Image &regularlyProcessedImage) = nullptr,
                   bool display = false);
    virtual ~ImageProcessor() {}

    virtual uint64_t outputTensorSizeInBytes();

    virtual DataElement operator()(DataElement &input);

   private:
    double minAspectRatio;
    double maxAspectRatio;
    double cropCenterToAspectRatio;
    uint32_t outputImageRows;
    uint32_t outputImageColumns;
    bool (*customProcessor)(Magick::Image &regularlyProcessedImage);
    bool display;

    std::mutex mutex;
};
