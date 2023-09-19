#include "Utilities/Common/Stream.h"

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
uint32_t maxNumGradientUpdateStreams = 32;

vector<deque<Stream>> uploadStreams;
mutex uploadStreamMutex;
uint32_t maxNumUploadStreams = 32;

vector<deque<Stream>> downloadStreams;
mutex downloadStreamMutex;
uint32_t maxNumDownloadStreams = 32;

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

void cleanUpHostFunctionArgs(Stream stream, HostFunctionArgsBase *args) {
    stream.synchronize();
    delete args;
}

void launchCleanUpHostFunctionArgs(Stream stream, HostFunctionArgsBase *args) {
    ThreadJoinQueue::instance().push(std::thread(&cleanUpHostFunctionArgs, stream, args));
}
