#include "Thor.h"

#include <boost/filesystem.hpp>

#include <assert.h>
#include <memory.h>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda_profiler_api.h>

using namespace ThorImplementation;
using namespace std;

int main() {
    Stream stream(0);

    vector<uint64_t> dimensions = {65536 * 16 * 16};
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(TensorDescriptor::DataType::UINT8, dimensions);

    Tensor t1(gpuPlacement, descriptor);
    Tensor t2(gpuPlacement, descriptor);
    Tensor t3(gpuPlacement, descriptor);

    for (uint32_t i = 0; i < 100; ++i)
        t1.subtract(t2, t3, stream);
    stream.synchronize();

    return 0;
}
