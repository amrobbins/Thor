#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include <assert.h>
#include <string>

namespace ThorImplementation {

class TensorPlacement {
   public:
    enum class MemDevices { INVALID = 0, CPU, GPU };

    TensorPlacement() { memDevice = MemDevices::INVALID; }
    explicit TensorPlacement(MemDevices memDevice, int deviceNum = 0) : memDevice(memDevice), deviceNum(deviceNum) {
        THOR_THROW_IF_FALSE(memDevice == MemDevices::CPU || memDevice == MemDevices::GPU);
        if (memDevice == MemDevices::CPU)
            THOR_THROW_IF_FALSE(deviceNum == 0);
    }

    MemDevices getMemDevice() const { return memDevice; }
    int getDeviceNum() const { return deviceNum; }

    bool operator==(const TensorPlacement& rhs) const { return memDevice == rhs.memDevice && deviceNum == rhs.deviceNum; }
    bool operator!=(const TensorPlacement& rhs) const { return !((*this) == rhs); }

    std::string toString() {
        THOR_THROW_IF_FALSE(memDevice == MemDevices::CPU || memDevice == MemDevices::GPU);
        std::string s;
        if (memDevice == MemDevices::CPU)
            return std::string("CPU");
        else
            return std::string("GPU:") + std::to_string(deviceNum);
    }

   private:
    MemDevices memDevice;
    int deviceNum;
};

}  // namespace ThorImplementation
