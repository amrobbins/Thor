#include "Utilities/Loaders/ShardedRawDatasetCreator.h"

using namespace boost::filesystem;

using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;

ShardedRawDatasetCreator::ShardedRawDatasetCreator(unordered_set<string> sourceDirectories,
                                                   unordered_set<string> destDirectories,
                                                   string baseDatasetFileName) {
    assert(!sourceDirectories.empty());
    assert(!destDirectories.empty());
    assert(!baseDatasetFileName.empty());

    this->sourceDirectories = sourceDirectories;
    this->destDirectories = destDirectories;
    this->baseDatasetFileName = baseDatasetFileName;

    numOutputShards = destDirectories.size();
    uint64_t i = 0;
    for (auto it = destDirectories.begin(); it != destDirectories.end(); ++it) {
        path shardPath(*it);
        shardPath /= (baseDatasetFileName + std::to_string(i));
        destShardFiles.push_back(shardPath);
        ++i;
    }
}

bool ShardedRawDatasetCreator::createDataset(unique_ptr<DataProcessor>&& dataProcessor) {
    uint32_t numOutputShards = destDirectories.size();
    uint64_t totalNumExamples = getNumExamples();
    uint64_t maxNumExamplesPerShard = (totalNumExamples / numOutputShards) + numOutputShards;
    uint64_t shardSizeInBytes = maxNumExamplesPerShard * dataProcessor->outputTensorSizeInBytes() + 1000000;

    // set up work queue
    uint32_t numProcessors = omp_get_num_procs();
    if (numProcessors > 1)
        numProcessors -= 1;
    unique_ptr<WorkQueueExecutorBase<DataElement, DataElement>> workQueueExecutor(
        (WorkQueueExecutorBase<DataElement, DataElement>*)dataProcessor.release());
    WorkQueueUnordered<DataElement, DataElement> workQueue;
    workQueue.open(workQueueExecutor, numProcessors, 32, 32);

    // Create the output mem-mapped file shards
    // FIXME
    printf("shardSizeInBytes %ld\n", shardSizeInBytes);

    // start numDestDisks threads, each pops a processed training example and writes it to the end of the memMappedFile that it is told too
    vector<thread> writerThreads;
    for (uint32_t i = 0; i < numOutputShards; ++i) {
        writerThreads.emplace_back(&ShardedRawDatasetCreator::writeDataToShard, this, &workQueue);
    }

    loadExamples(workQueue);

    // wait for processing to finish
    while (!workQueue.isEmpty()) {
        usleep(250000);
    }

    workQueue.close();

    // join each writer thread
    while (!writerThreads.empty()) {
        writerThreads.back().join();
        writerThreads.pop_back();
    }

    return true;
}

uint64_t ShardedRawDatasetCreator::getNumExamples() {
    uint64_t totalNumExamples = 0;

    for (auto it = sourceDirectories.begin(); it != sourceDirectories.end(); ++it) {
        string datasetDirectoryString = *it;

        for (int rawType = (int)ExampleType::TRAIN; rawType <= (int)ExampleType::TEST; ++rawType) {
            ExampleType type = (ExampleType)rawType;

            string exampleType;
            if (type == ExampleType::TRAIN)
                exampleType = "train";
            else if (type == ExampleType::VALIDATE)
                exampleType = "validate";
            else if (type == ExampleType::TEST)
                exampleType = "test";
            else
                assert(false);

            uint32_t numProcessors = omp_get_num_procs();
            if (numProcessors > 1)
                numProcessors -= 1;

            omp_set_num_threads(numProcessors);
            vector<vector<path>> classesPerThread(numProcessors);

            path datasetDirectory = datasetDirectoryString;
            datasetDirectory /= exampleType;
            printf("%s\n", datasetDirectory.native().c_str());
            fflush(stdout);
            assert(is_directory(datasetDirectory));
            uint32_t thread = 0;
            for (directory_entry& classDirectory : directory_iterator(datasetDirectory)) {
                if (is_directory(classDirectory.path()) && !classDirectory.path().filename_is_dot() &&
                    !classDirectory.path().filename_is_dot_dot()) {
                    classesPerThread[thread].push_back(classDirectory.path());
                    thread = (thread + 1) % numProcessors;
                }
            }

            uint64_t numExamples = 0;
#pragma omp parallel for schedule(static, 1) reduction(+ : numExamples)
            for (uint64_t processor = 0; processor < numProcessors; ++processor) {
                uint64_t numClasses = classesPerThread[processor].size();
                for (uint64_t i = 0; i < numClasses; ++i) {
                    path classDirectory = classesPerThread[processor][i];
                    bool addToClasses = (type == ExampleType::TRAIN);
                    for (directory_entry& example : directory_iterator(classDirectory)) {
                        if (is_regular_file(example.path())) {
                            numExamples += 1;
                            if (addToClasses) {
                                mtx.lock();
                                string className = classDirectory.filename().native();
                                classes.insert(className);
                                mtx.unlock();
                                addToClasses = false;
                            }
                        }
                    }
                }
            }
            totalNumExamples += numExamples;
        }
    }
    return totalNumExamples;
}

void ShardedRawDatasetCreator::loadExamples(WorkQueueUnordered<DataElement, DataElement>& workQueue) {
    assert(!sourceDirectories.empty());
    omp_set_num_threads(sourceDirectories.size());

    vector<string> sourceDirectoriesVector(sourceDirectories.begin(), sourceDirectories.end());
    uint32_t numSourceDirectories = sourceDirectoriesVector.size();

#pragma omp parallel for schedule(static, 1)
    for (uint32_t i = 0; i < numSourceDirectories; ++i) {
        string datasetDirectoryString = sourceDirectoriesVector[i];
        uint64_t destShard = 0;

        for (int rawType = (int)ExampleType::TRAIN; rawType <= (int)ExampleType::TEST; ++rawType) {
            ExampleType type = (ExampleType)rawType;

            string exampleType;
            if (type == ExampleType::TRAIN)
                exampleType = "train";
            else if (type == ExampleType::VALIDATE)
                exampleType = "validate";
            else if (type == ExampleType::TEST)
                exampleType = "test";
            else
                assert(false);

            path datasetDirectory = datasetDirectoryString;
            datasetDirectory /= exampleType;
            assert(is_directory(datasetDirectory));

            for (directory_entry& classDirectory : directory_iterator(datasetDirectory)) {
                path classDirectoryPath = classDirectory.path();
                if (is_directory(classDirectoryPath) && !classDirectoryPath.filename_is_dot() &&
                    !classDirectoryPath.filename_is_dot_dot()) {
                    string className = classDirectoryPath.filename().native();
                    vector<path> examples;
                    if (classes.count(className) == 0)
                        continue;

                    for (directory_entry& example : directory_iterator(classDirectoryPath)) {
                        if (is_regular_file(example.path())) {
                            path examplePath = example.path();
                            std::ifstream file(examplePath.native(), std::ios::binary | std::ios::ate);
                            uint64_t fileSizeInBytes = file.tellg();

                            // read file
                            file.seekg(0, std::ios::beg);
                            unique_ptr<char> buffer(new char[fileSizeInBytes]);
                            if (file.read(buffer.get(), fileSizeInBytes)) {
                                DataElement rawDataElement;
                                rawDataElement.data.reset(buffer.release());
                                rawDataElement.numDataBytes = fileSizeInBytes;
                                rawDataElement.exampleType = type;
                                rawDataElement.className = className;
                                rawDataElement.destShard = destShardFiles[destShard];
                                destShard = (destShard + 1) % numOutputShards;

                                printf("pushing %s to %s\n", rawDataElement.className.c_str(), rawDataElement.destShard.native().c_str());

                                bool success = workQueue.push(rawDataElement);
                                assert(success);
                            } else {
                                assert(false);
                            }
                        }
                    }
                }
            }
        }
    }
}

void ShardedRawDatasetCreator::writeDataToShard(WorkQueueUnordered<DataElement, DataElement>* workQueue) {
    // FIXME
    bool queueOpen = true;
    DataElement rawDataElement;
    string type;
    while (queueOpen) {
        queueOpen = workQueue->pop(rawDataElement);
        type = rawDataElement.exampleType == ExampleType::TRAIN
                   ? "train"
                   : (rawDataElement.exampleType == ExampleType::VALIDATE ? "validate" : "test");
        printf(
            "writing %s class %s to shard %s\n", type.c_str(), rawDataElement.className.c_str(), rawDataElement.destShard.native().c_str());
    }
    printf("queue closed\n");
}
