#include "Utilities/Loaders/ImageProcessor.h"

using namespace std;

ImageProcessor::ImageProcessor(double minAspectRatio,
                               double maxAspectRatio,
                               uint32_t outputImageRows,
                               uint32_t outputImageColumns,
                               uint32_t bytesPerPixelChannel,
                               bool centerAndScale,
                               bool display) {
    this->bytesPerPixelChannel = bytesPerPixelChannel;
    if (centerAndScale)
        assert(bytesPerPixelChannel == 2);

    this->minAspectRatio = minAspectRatio;
    this->maxAspectRatio = maxAspectRatio;
    this->outputImageRows = outputImageRows;
    this->outputImageColumns = outputImageColumns;
    this->centerAndScale = centerAndScale;
    this->display = display;
    customProcessorUint8 = nullptr;
    customProcessorHalf = nullptr;
}

ImageProcessor::ImageProcessor(double minAspectRatio,
                               double maxAspectRatio,
                               uint32_t outputImageRows,
                               uint32_t outputImageColumns,
                               bool (*customProcessor)(uint8_t *regularlyProcessedImage),
                               bool display) {
    bytesPerPixelChannel = 1;
    this->minAspectRatio = minAspectRatio;
    this->maxAspectRatio = maxAspectRatio;
    this->outputImageRows = outputImageRows;
    this->outputImageColumns = outputImageColumns;
    this->centerAndScale = false;
    this->display = display;
    customProcessorUint8 = customProcessor;
    customProcessorHalf = nullptr;
}

ImageProcessor::ImageProcessor(double minAspectRatio,
                               double maxAspectRatio,
                               uint32_t outputImageRows,
                               uint32_t outputImageColumns,
                               bool (*customProcessor)(half *regularlyProcessedImage),
                               bool centerAndScale,
                               bool display) {
    bytesPerPixelChannel = 2;
    this->minAspectRatio = minAspectRatio;
    this->maxAspectRatio = maxAspectRatio;
    this->outputImageRows = outputImageRows;
    this->outputImageColumns = outputImageColumns;
    this->centerAndScale = centerAndScale;
    this->display = display;
    customProcessorUint8 = nullptr;
    customProcessorHalf = customProcessor;
}

uint64_t ImageProcessor::outputTensorSizeInPixels() { return outputImageRows * outputImageColumns; }
uint64_t ImageProcessor::outputTensorSizeInBytes() { return 3 * outputTensorSizeInPixels() * bytesPerPixelChannel; }

ThorImplementation::TensorDescriptor::DataType ImageProcessor::getDataType() {
    if (bytesPerPixelChannel == 1)
        return ThorImplementation::TensorDescriptor::DataType::UINT8;
    else
        return ThorImplementation::TensorDescriptor::DataType::FP16;
}

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

    if (bytesPerPixelChannel == 1) {
        output.dataType = ThorImplementation::TensorDescriptor::DataType::UINT8;

        success = ImageLoader::toRgbArray(image, data.get(), ImageLoader::Layout::CHW);
        if (!success)
            return output;

        if (customProcessorUint8 != nullptr) {
            bool useImage = customProcessorUint8(data.get());
            if (!useImage)
                // In this case, output is a copy of input but with a null data pointer
                return output;
        }
    } else {
        output.dataType = ThorImplementation::TensorDescriptor::DataType::FP16;

        success = ImageLoader::toRgbArray(image, (half *)data.get(), ImageLoader::Layout::CHW, centerAndScale);
        if (!success)
            return output;

        if (customProcessorHalf != nullptr) {
            bool useImage = customProcessorHalf((half *)data.get());
            if (!useImage)
                // In this case, output is a copy of input but with a null data pointer
                return output;
        }
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
