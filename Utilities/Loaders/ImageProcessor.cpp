#include "Utilities/Loaders/ImageProcessor.h"

using namespace std;

ImageProcessor::ImageProcessor(double minAspectRatio,
                               double maxAspectRatio,
                               uint32_t outputImageRows,
                               uint32_t outputImageColumns,
                               bool (*customProcessor)(uint8_t *regularlyProcessedImage),
                               bool display) {
    this->minAspectRatio = minAspectRatio;
    this->maxAspectRatio = maxAspectRatio;
    this->outputImageRows = outputImageRows;
    this->outputImageColumns = outputImageColumns;
    this->customProcessor = customProcessor;
    this->display = display;
}

uint64_t ImageProcessor::outputTensorSizeInBytes() { return 3 * outputImageRows * outputImageColumns; }

DataElement ImageProcessor::operator()(DataElement &input) {
    bool success;
    Magick::Image image;
    DataElement output;

    output = input;
    output.data = nullptr;
    output.numDataBytes = 0;

    success = ImageLoader::loadImage(input.data.get(), input.numDataBytes, image);
    if (!success)
        return output;
    success = ImageLoader::resizeImage(minAspectRatio, maxAspectRatio, outputImageRows, outputImageColumns, image);
    if (!success)
        return output;

    unique_ptr<uint8_t> data(new uint8_t[outputTensorSizeInBytes()]);
    success = ImageLoader::toRgbArray(image, data.get(), true);
    if (!success)
        return output;

    if (customProcessor != nullptr) {
        bool useImage = customProcessor(data.get());
        if (!useImage)
            return output;
    }

    if (display) {
        mutex.lock();
        image.display();
        mutex.unlock();
    }

    output.numDataBytes = outputTensorSizeInBytes();
    output.data.reset(data.release());
    return output;
}
