#include "Thor.h"

using namespace Thor;

int main() {
    uint32_t gpuNum = 0;
    uint32_t numStamps = 1;
    uint32_t batchSize = 256;
    uint32_t numBatches = 250;
    HackathonExecutor(buildAlexNet(), gpuNum, numStamps, batchSize, numBatches);

    return 0;
}
