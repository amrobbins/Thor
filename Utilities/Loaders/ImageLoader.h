#pragma once

#include <Magick++.h>

#include <assert.h>
#include <math.h>

// using namespace Magick;

class ImageLoader {
   public:
    // static bool loadImage(const std::string &filename, Magick::Image &loadedImage);
    static bool loadImage(const std::string &filename, Magick::Image &image);

    // Load image from the raw contents of a file loaded into memory:
    static bool loadImage(void *rawImageData, uint64_t rawImageDataSizeInBytes, Magick::Image &image);

    static bool resizeImage(
        double minAspectRatio, double maxAspectRatio, uint32_t outputImageRows, uint32_t outputImageColumns, Magick::Image &image);

    // Returns 1 byte for each of rgb in CHW format, for example [r0, g0, b0, r1, g1, b1, ...]
    static bool toRgbArray(Magick::Image &image, uint8_t *rgbPixelArray, bool toHWCLayout);

   private:
    ImageLoader() {}
    static void convertRGB_HWCtoCHW(uint8_t *sourceRgbPixelArray, uint8_t *destRgbPixelArray, uint64_t rows, uint64_t cols);

    struct MagickInitializer {
        MagickInitializer();
    };
    static MagickInitializer magickInitializer;
};
