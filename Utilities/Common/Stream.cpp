#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"

using namespace std;

int Stream::numCudnnHandles = 0;
int Stream::numCublasHandles = 0;

// Note: These are global because destroying a stream when static members are destroyed seems to be a problem.
// Also Note: I would rather be able to use unlimited streams to avoid potential false dependencies in very large very branched networks
//            But I need to support whatever hardware limitation that may exist, so I have the ability to set lower limits on the number
//            of streams that are in place. I don't do this for forward/backward (i.e. data) streams because false dependencies along
//            the execution graph could result in deadlock.
vector<deque<Stream>> gradientUpdateStreams;
mutex gradientUpdateStreamMutex;
uint32_t maxNumGradientUpdateStreams = 8;

vector<deque<Stream>> uploadStreams;
mutex uploadStreamMutex;
uint32_t maxNumUploadStreams = 4;

vector<deque<Stream>> downloadStreams;
mutex downloadStreamMutex;
uint32_t maxNumDownloadStreams = 4;

// To allow for parallelization while limiting the amount of streams created, gradient update operations all share
// a fixed size pool of gradientUpdateStreams
Stream Stream::getNextGradientUpdateStream(uint32_t deviceNum) {
    unique_lock<mutex> lck(gradientUpdateStreamMutex);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        assert(deviceNum < numGpus);
    }

    assert(maxNumGradientUpdateStreams > 0);

    while (gradientUpdateStreams.size() < numGpus)
        gradientUpdateStreams.emplace_back();

    // I never delete streams since they may be in use. Only ever add new ones.
    if (gradientUpdateStreams[deviceNum].size() < maxNumGradientUpdateStreams) {
        gradientUpdateStreams[deviceNum].emplace_front(deviceNum);
        gradientUpdateStreams[deviceNum].front().informIsStatic();
    }

    Stream stream = gradientUpdateStreams[deviceNum].front();
    gradientUpdateStreams[deviceNum].pop_front();
    gradientUpdateStreams[deviceNum].push_back(stream);
    return stream;
}

// When you need one for maybe a single use and do not want to advance the round-robin sequence.
Stream Stream::getMostRecentGradientUpdateStream(uint32_t deviceNum) {
    unique_lock<mutex> lck(gradientUpdateStreamMutex);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        assert(deviceNum < numGpus);
    }

    assert(maxNumGradientUpdateStreams > 0);

    while (gradientUpdateStreams.size() < numGpus)
        gradientUpdateStreams.emplace_back();

    if (gradientUpdateStreams[deviceNum].empty()) {
        gradientUpdateStreams[deviceNum].emplace_front(deviceNum);
        gradientUpdateStreams[deviceNum].front().informIsStatic();
    }

    Stream stream = gradientUpdateStreams[deviceNum].front();
    return stream;
}

void Stream::setMaxNumGradientUpdateStreams(uint32_t numGradientUpdateStreams) {
    unique_lock<mutex> lck(gradientUpdateStreamMutex);

    maxNumGradientUpdateStreams = numGradientUpdateStreams;
}

Stream Stream::getNextUploadStream(uint32_t deviceNum) {
    unique_lock<mutex> lck(uploadStreamMutex);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        assert(deviceNum < numGpus);
    }

    assert(maxNumUploadStreams > 0);

    while (uploadStreams.size() < numGpus)
        uploadStreams.emplace_back();

    // I never delete streams since they may be in use. Only ever add new ones.
    if (uploadStreams[deviceNum].size() < maxNumUploadStreams) {
        uploadStreams[deviceNum].emplace_front(deviceNum);
        uploadStreams[deviceNum].front().informIsStatic();
    }

    Stream stream = uploadStreams[deviceNum].front();
    uploadStreams[deviceNum].pop_front();
    uploadStreams[deviceNum].push_back(stream);
    return stream;
}

void Stream::setMaxNumUploadStreams(uint32_t numUploadStreams) {
    unique_lock<mutex> lck(uploadStreamMutex);

    maxNumUploadStreams = numUploadStreams;
}

Stream Stream::getNextDownloadStream(uint32_t deviceNum) {
    unique_lock<mutex> lck(downloadStreamMutex);

    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    if (deviceNum >= numGpus) {
        printf("Error: trying to get a stream for gpu %d but there are only %d gpus\n", deviceNum, numGpus);
        fflush(stdout);
        assert(deviceNum < numGpus);
    }

    assert(maxNumDownloadStreams > 0);

    while (downloadStreams.size() < numGpus)
        downloadStreams.emplace_back();

    // I never delete streams since they may be in use. Only ever add new ones.
    if (downloadStreams[deviceNum].size() < maxNumDownloadStreams) {
        downloadStreams[deviceNum].emplace_front(deviceNum);
        downloadStreams[deviceNum].front().informIsStatic();
    }

    Stream stream = downloadStreams[deviceNum].front();
    downloadStreams[deviceNum].pop_front();
    downloadStreams[deviceNum].push_back(stream);
    return stream;
}

void Stream::setMaxNumDownloadStreams(uint32_t numDownloadStreams) {
    unique_lock<mutex> lck(downloadStreamMutex);

    maxNumDownloadStreams = numDownloadStreams;
}

void cleanUpHostFunctionArgs(Stream stream, unique_ptr<HostFunctionArgsBase> &&args) {
    stream.synchronize();
    // Now args go out of scope and is deleted
}

void Stream::launchCleanUpHostFunctionArgs(unique_ptr<HostFunctionArgsBase> &&args) {
    ThreadJoinQueue::instance().push(thread(&cleanUpHostFunctionArgs, *this, std::move(args)));
    assert(args == nullptr);
}

void Stream::waitEvent(Event event) const {
    assert(!uninitialized());

    ScopedGpu scopedGpu(gpuNum);

    CUDA_CHECK(cudaStreamWaitEvent(cudaStream, event.getEvent(), 0));
}

void Stream::synchronize() const {
    assert(!uninitialized());

    CUDA_CHECK(cudaStreamSynchronize(cudaStream));
}

void Stream::deviceSynchronize(int gpuNum) {
    ScopedGpu scopedGpu(gpuNum);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Stream::enqueueHostFunction(cudaHostFn_t function, std::unique_ptr<HostFunctionArgsBase> &&args) {
    CUDA_CHECK(cudaLaunchHostFunc(*this, function, args.get()));
    launchCleanUpHostFunctionArgs(std::move(args));
}

void Stream::construct(int gpuNum, Priority priority) {
    ReferenceCounted::initialize();

    cudnnHandle = new Optional<cudnnHandle_t>;
    cublasHandle = new Optional<cublasHandle_t>;
    mtx = new std::mutex;

    ScopedGpu scopedGpu(gpuNum);
    this->gpuNum = gpuNum;

    // greatestPriority is given the highest priority in terms of execution, and its numerical value is the minimum of the allowed
    // range.
    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    int priorityValue;
    if (priority == Priority::HIGH)
        priorityValue = greatestPriority;
    else if (priority == Priority::REGULAR)
        priorityValue = greatestPriority + 1;
    else
        priorityValue = greatestPriority + 2;

    CUDA_CHECK(cudaStreamCreateWithPriority(&cudaStream, cudaStreamNonBlocking, priorityValue));
}

void Stream::destroy() {
    // If this is a static stream, it is too late to free cuda resources, or interact with cuda, when it is destroyed.
    // Doesn't much matter as the program is exiting anyway.
    if (!isStatic) {
        ScopedGpu scopedGpu(gpuNum);

        // can't destroy the cudnn handle at the point when the static string is destroyed
        if (cudnnHandle->isPresent()) {
            numCudnnHandles -= 1;

            cudnnStatus_t cudnnStatus;
            cudnnStatus = cudnnDestroy(*cudnnHandle);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        }

        if (cublasHandle->isPresent()) {
            numCublasHandles -= 1;

            cublasStatus_t cublasStatus;
            cublasStatus = cublasDestroy(*cublasHandle);
            assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
        }

        CUDA_CHECK(cudaStreamDestroy(cudaStream));

        delete cudnnHandle;
        cudnnHandle = nullptr;
        delete cublasHandle;
        cublasHandle = nullptr;
    }
    delete mtx;
    mtx = nullptr;
}
