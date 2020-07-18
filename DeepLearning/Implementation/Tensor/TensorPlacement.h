#pragma once

#include <assert.h>
#include <string>

using std::string;

class TensorPlacement {
   public:
    enum class MemDevices { INVALID = 0, CPU, GPU };

    TensorPlacement() { memDevice = MemDevices::INVALID; }
    TensorPlacement(MemDevices memDevice, int deviceNum = 0) : memDevice(memDevice), deviceNum(deviceNum) {
        assert(memDevice == MemDevices::CPU || memDevice == MemDevices::GPU);
        if (memDevice == MemDevices::CPU)
            assert(deviceNum == 0);
    }

    MemDevices getMemDevice() { return memDevice; }
    int getDeviceNum() { return deviceNum; }

    bool operator==(const TensorPlacement& rhs) { return memDevice == rhs.memDevice && deviceNum == rhs.deviceNum; }
    bool operator!=(const TensorPlacement& rhs) { return !((*this) == rhs); }

    string toString() {
        assert(memDevice == MemDevices::CPU || memDevice == MemDevices::GPU);
        string s;
        if (memDevice == MemDevices::CPU)
            return string("CPU");
        else
            return string("GPU:") + std::to_string(deviceNum);
    }

   private:
    MemDevices memDevice;
    int deviceNum;
};
