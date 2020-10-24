#include "Utilities/Common/CudnnHelper.h"

using namespace ThorImplementation;

std::map<uint32_t, std::map<std::thread::id, cudnnHandle_t>> CudnnHelper::cudnnHandles;

std::mutex CudnnHelper::mtx;
