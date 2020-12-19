#include "Utilities/Loaders/ImageLoader.h"
#include <X11/Xlib.h>

using namespace Magick;
using namespace std;

bool ImageLoader::loadImage(const string &filename, Image &image) {
    try {
        image = Image(filename);
    } catch (Exception &e) {
        return false;
    }

    return true;
}

bool ImageLoader::loadImage(void *rawImageData, uint64_t rawImageDataSizeInBytes, Image &image) {
    try {
        Blob imageBlob(rawImageData, rawImageDataSizeInBytes);
        image = Image(imageBlob);
    } catch (Exception &e) {
        return false;
    }

    return true;
}

bool ImageLoader::resizeImage(
    double minAspectRatio, double maxAspectRatio, uint32_t outputImageRows, uint32_t outputImageColumns, Image &image) {
    if (minAspectRatio > 0 && maxAspectRatio > 0)
        assert(minAspectRatio <= maxAspectRatio);

    double rowAspectRatio = (double)outputImageRows / image.rows();
    double colAspectRatio = (double)outputImageColumns / image.columns();
    if (minAspectRatio > 0.0 && (rowAspectRatio < minAspectRatio || colAspectRatio < minAspectRatio))
        return false;
    if (maxAspectRatio > 0.0 && (rowAspectRatio > maxAspectRatio || colAspectRatio > maxAspectRatio))
        return false;

    if (image.rows() != outputImageRows || image.columns() != outputImageColumns) {
        // zoom
        double aspectRatio = max(rowAspectRatio, colAspectRatio);
        uint32_t numResizedRows = ceil(image.rows() * aspectRatio);
        uint32_t numResizedColumns = ceil(image.columns() * aspectRatio);
        image.resize(Geometry(numResizedColumns, numResizedRows));

        assert(numResizedRows >= outputImageRows);
        assert(numResizedColumns >= outputImageColumns);
        if (numResizedRows > outputImageRows || numResizedColumns > outputImageColumns) {
            // crop
            uint32_t cropStartRow = (numResizedRows - outputImageRows) / 2;
            uint32_t cropStartCol = (numResizedColumns - outputImageColumns) / 2;
            // Geometry(size_t width_, size_t height_, ssize_t xOff_ = 0, ssize_t yOff_ = 0)
            image.crop(Geometry(outputImageColumns, outputImageRows, cropStartCol, cropStartRow));
            return true;
        }
    }

    return true;
}

bool ImageLoader::toRgbArray(Image &image, uint8_t *rgbPixelArray, bool toHWCLayout) {
    try {
        if (toHWCLayout) {
            image.getConstPixels(0, 0, image.columns(), image.rows());
            image.writePixels(RGBQuantum, rgbPixelArray);
        } else {
            uint8_t *buffer = new uint8_t[3 * image.rows() * image.columns()];
            image.getConstPixels(0, 0, image.columns(), image.rows());
            image.writePixels(RGBQuantum, buffer);
            convertRGB_HWCtoCHW(buffer, rgbPixelArray, image.rows(), image.columns());
            delete[] buffer;
        }
    } catch (Exception &e) {
        return false;
    }
    return true;
}

void ImageLoader::convertRGB_HWCtoCHW(uint8_t *sourceRgbPixelArray, uint8_t *destRgbPixelArray, uint64_t rows, uint64_t cols) {
    uint64_t columnBytes = 3 * cols;
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c) {
            destRgbPixelArray[r * cols + c] = sourceRgbPixelArray[r * columnBytes + 3 * c];
            destRgbPixelArray[r * cols + c + (rows * cols)] = sourceRgbPixelArray[r * columnBytes + 3 * c + 1];
            destRgbPixelArray[r * cols + c + 2 * (rows * cols)] = sourceRgbPixelArray[r * columnBytes + 3 * c + 2];
        }
    }
}

ImageLoader::MagickInitializer::MagickInitializer() {
    XInitThreads();
    InitializeMagick(nullptr);
}
ImageLoader::MagickInitializer ImageLoader::magickInitializer;
