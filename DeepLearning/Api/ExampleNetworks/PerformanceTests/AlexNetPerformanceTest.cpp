#include "Thor.h"

using namespace Thor;

int main() {
    uint32_t gpuNum = 0;
    uint32_t numStamps = 3;
    uint32_t batchSize = 128;
    uint32_t numBatches = 150;
    HackathonExecutor(buildAlexNet(), gpuNum, numStamps, batchSize, numBatches);

    return 0;
}
