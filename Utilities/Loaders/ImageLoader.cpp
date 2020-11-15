#include "Utilities/Loaders/ImageLoader.h"

using namespace Magick;
using namespace std;

bool ImageLoader::loadImage(const string &filename, Image &image) {
    try {
        image = Image(filename);
    } catch (Exception e) {
        return false;
    }

    return true;
}

bool ImageLoader::loadImage(void *rawImageData, uint64_t rawImageDataSizeInBytes, Image &image) {
    try {
        Blob imageBlob(rawImageData, rawImageDataSizeInBytes);
        image = Image(imageBlob);
    } catch (Exception e) {
        return false;
    }

    return true;
}

bool ImageLoader::resizeImage(double minAspectRatio,
                              double maxAspectRatio,
                              double cropCenterToAspectRatio,
                              uint32_t outputImageRows,
                              uint32_t outputImageColumns,
                              Image &image) {
    if (minAspectRatio > 0 && maxAspectRatio > 0)
        assert(minAspectRatio <= maxAspectRatio);
    if (cropCenterToAspectRatio > 0)
        assert(cropCenterToAspectRatio >= 1.0);
    if (cropCenterToAspectRatio > 0 && minAspectRatio > 0)
        assert(cropCenterToAspectRatio >= minAspectRatio);
    if (cropCenterToAspectRatio > 0 && maxAspectRatio > 0)
        assert(cropCenterToAspectRatio <= maxAspectRatio);

    double rowAspectRatio = outputImageRows / image.rows();
    double colAspectRatio = outputImageColumns / image.columns();
    if (minAspectRatio > 0.0 && (rowAspectRatio < minAspectRatio || colAspectRatio < minAspectRatio))
        return false;
    if (maxAspectRatio > 0.0 && (rowAspectRatio > maxAspectRatio || colAspectRatio < maxAspectRatio))
        return false;

    if (rowAspectRatio > cropCenterToAspectRatio || colAspectRatio > cropCenterToAspectRatio) {
        // crop
        uint32_t numCroppedRows = image.rows();
        uint32_t numCroppedColumns = image.columns();
        if (rowAspectRatio > cropCenterToAspectRatio)
            numCroppedRows = ceil(cropCenterToAspectRatio * image.rows());
        if (colAspectRatio > cropCenterToAspectRatio)
            numCroppedColumns = ceil(cropCenterToAspectRatio * image.columns());
        uint32_t cropStartRow = (numCroppedRows - image.rows()) / 2;
        uint32_t cropStartCol = (numCroppedColumns - image.columns()) / 2;
        image.crop(Geometry(numCroppedColumns, numCroppedRows, cropStartCol, cropStartRow));
        rowAspectRatio = outputImageRows / image.rows();
        colAspectRatio = outputImageColumns / image.columns();
    }

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

bool ImageLoader::toRgbArray(Image &image, uint8_t *rgbPixelArray) {
    try {
        image.writePixels(RGBQuantum, rgbPixelArray);
    } catch (Exception e) {
        return false;
    }
    return true;
}
