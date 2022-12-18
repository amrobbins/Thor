#include "Utilities/Common/Stream.h"

int Stream::numCudnnHandles = 0;
std::unordered_map<uint32_t, Stream> Stream::staticDeviceStreams;

// For cases where you just need any stream (for example computing workspace sizes for cudnn) use the static stream to
// avoid allocating too many streams and crashing Cuda.
Stream Stream::getStaticStream(uint32_t deviceNum) {
    assert(deviceNum < MachineEvaluator::instance().getNumGpus());
    if (staticDeviceStreams.count(deviceNum) == 0) {
        Stream stream(deviceNum);
        stream.informIsStatic();
        staticDeviceStreams[deviceNum] = stream;
    }
    return staticDeviceStreams[deviceNum];
}
