#include "Utilities/Loaders/ShardedRawDatasetCreator.h"

using std::make_pair;
using std::make_shared;
using std::map;
using std::mutex;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;

using namespace std::filesystem;

ShardedRawDatasetCreator::ShardedRawDatasetCreator(unordered_set<string> sourceDirectories,
                                                   unordered_set<string> destDirectories,
                                                   string baseDatasetFileName,
                                                   uint32_t maxClasses)
    : maxClasses(maxClasses) {
    assert(!sourceDirectories.empty());
    assert(!destDirectories.empty());
    assert(!baseDatasetFileName.empty());

    this->sourceDirectories = sourceDirectories;
    this->destDirectories = destDirectories;
    this->baseDatasetFileName = baseDatasetFileName;

    destShardTrain = 0;
    destShardValidate = 0;
    destShardTest = 0;

    numOutputShards = destDirectories.size();
    uint64_t i = 0;
    for (auto it = destDirectories.begin(); it != destDirectories.end(); ++it) {
        path shardPath(*it);
        if (maxClasses > 0)
            shardPath /= (baseDatasetFileName + "_" + std::to_string(maxClasses) + "_classes_" + std::to_string(i + 1) + "_of_" +
                          std::to_string(destDirectories.size())) +
                         ".shard";
        else
            shardPath /= (baseDatasetFileName + "_" + std::to_string(i + 1) + "_of_" + std::to_string(destDirectories.size())) + ".shard";
        destShardFiles.push_back(shardPath);
        ++i;
    }
}

bool ShardedRawDatasetCreator::createDataset(unique_ptr<DataProcessor>&& dataProcessor, std::vector<std::shared_ptr<Shard>>& shards) {
    uint32_t numOutputShards = destDirectories.size();
    uint64_t numTrainExamples, numValidateExamples, numTestExamples;
    uint64_t maxFilenameChars;
    uint64_t maxClassNameChars;
    set<string> allClasses;
    getNumExamples(numTrainExamples, numValidateExamples, numTestExamples, maxFilenameChars, allClasses, maxClassNameChars);
    uint64_t numTrainExamplesPerShard = (numTrainExamples / numOutputShards) + numOutputShards;
    uint64_t numValidateExamplesPerShard = (numValidateExamples / numOutputShards) + numOutputShards;
    uint64_t numTestExamplesPerShard = (numTestExamples / numOutputShards) + numOutputShards;
    outputTensorSizeInBytes = dataProcessor->outputTensorSizeInBytes();
    ThorImplementation::TensorDescriptor::DataType dataType = dataProcessor->getDataType();

    // set up work queue
    uint32_t numProcessors = omp_get_num_procs();
    if (numProcessors > 1)
        numProcessors -= 1;
    unique_ptr<WorkQueueExecutorBase<DataElement, DataElement>> workQueueExecutor(
        (WorkQueueExecutorBase<DataElement, DataElement>*)dataProcessor.release());
    WorkQueueUnordered<DataElement, DataElement> workQueue;
    workQueue.open(workQueueExecutor, numProcessors, 32, 32);

    vector<string> allClassesVector;
    allClassesVector.reserve(allClasses.size());
    for (auto it = allClasses.begin(); it != allClasses.end(); ++it) {
        allClassesVector.push_back(*it);
    }

    // Create the output mem-mapped file shards
    for (uint32_t i = 0; i < destShardFiles.size(); ++i) {
        shards.push_back(make_shared<Shard>());
        shards.back()->createShard(destShardFiles[i].native(),
                                   numTrainExamplesPerShard,
                                   numValidateExamplesPerShard,
                                   numTestExamplesPerShard,
                                   outputTensorSizeInBytes,
                                   dataType,
                                   maxFilenameChars,
                                   allClassesVector,
                                   maxClassNameChars);
    }

    // It goes like this:
    // A work queue is created that will send the file's bytes through the data processor upon the file's buffer being pushed onto the work
    // queue.
    // loadExamples is the function that pushes the buffers onto the work queue, which is why it needs to be passed the work queue, it
    // reads the files from disk and loads them into a buffer and then pushes them onto the work queue
    // writeDataToShard pops the processed buffer from the work queue and each shard is appended to round-robin regardless of which
    // thread does the appending.
    // loadExamples returns after all examples are loaded. Then we wait for all the buffers to be popped from the work queue.
    // Then we close the queue, which causes the writer threads to terminate.

    // start numDestDisks threads, each pops a processed training example and writes it to the end of the memMappedFile that it is told too
    std::vector<thread> writerThreads;
    for (uint32_t i = 0; i < numOutputShards; ++i) {
        writerThreads.emplace_back(&ShardedRawDatasetCreator::writeDataToShard, this, &workQueue, &shards);
    }

    loadExamples(workQueue);

    // wait for processing to finish
    while (!workQueue.isEmpty()) {
        usleep(250000);
    }

    workQueue.close();

    // join each writer thread
    // mutex is used to ensure each writer thread has finished, before shrinking the shard.
    mutex mtx;
    mtx.lock();
    while (!writerThreads.empty()) {
        writerThreads.back().join();
        writerThreads.pop_back();
    }
    mtx.unlock();

    for (uint32_t i = 0; i < shards.size(); ++i) {
        shards[i]->shrinkToFit();
    }

    return true;
}

void ShardedRawDatasetCreator::getNumExamples(uint64_t& numTrainExamples,
                                              uint64_t& numValidateExamples,
                                              uint64_t& numTestExamples,
                                              uint64_t& maxFilenameChars,
                                              set<string>& allClasses,
                                              uint64_t& maxClassNameChars) {
    numTrainExamples = 0;
    numValidateExamples = 0;
    numTestExamples = 0;

    maxFilenameChars = 0;
    maxClassNameChars = 0;

    for (auto it = sourceDirectories.begin(); it != sourceDirectories.end(); ++it) {
        string datasetDirectoryString = *it;

        // When choosing some max number of classes, first figure out what classes will be chosen:
        if (maxClasses != 0) {
            path datasetDirectory = datasetDirectoryString;
            datasetDirectory /= "train";
            assert(is_directory(datasetDirectory));
            for (const directory_entry& classDirectory : directory_iterator(datasetDirectory)) {
                string filename = classDirectory.path().filename();
                bool filename_is_dot = filename == ".";
                bool filename_is_dot_dot = filename == "..";
                if (is_directory(classDirectory.path()) && !filename_is_dot && !filename_is_dot_dot) {
                    if (allClasses.size() == maxClasses)
                        break;
                    allClasses.insert(classDirectory.path().filename().native());
                }
            }
        }

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
            std::vector<std::vector<path>> classesPerThread(numProcessors);

            path datasetDirectory = datasetDirectoryString;
            datasetDirectory /= exampleType;
            assert(is_directory(datasetDirectory));
            uint32_t thread = 0;
            for (const directory_entry& classDirectory : directory_iterator(datasetDirectory)) {
                string filename = classDirectory.path().filename();
                bool filename_is_dot = filename == ".";
                bool filename_is_dot_dot = filename == "..";
                if (is_directory(classDirectory.path()) && !filename_is_dot && !filename_is_dot_dot) {
                    if (maxClasses != 0 && allClasses.count(classDirectory.path().filename().native()) == 0)
                        continue;
                    classesPerThread[thread].push_back(classDirectory.path());
                    allClasses.insert(classDirectory.path().filename().native());
                    thread = (thread + 1) % numProcessors;
                }
            }

            uint64_t numExamples = 0;
#pragma omp parallel for schedule(static, 1) reduction(+ : numExamples) reduction(max : maxFilenameChars)
            for (uint64_t processor = 0; processor < numProcessors; ++processor) {
                uint64_t numClasses = classesPerThread[processor].size();
                for (uint64_t i = 0; i < numClasses; ++i) {
                    path classDirectory = classesPerThread[processor][i];
                    bool addToClasses = (type == ExampleType::TRAIN);
                    for (const directory_entry& example : directory_iterator(classDirectory)) {
                        if (is_regular_file(example.path())) {
                            string filename = example.path().filename().native();
                            string className = classDirectory.filename().native();
                            if (filename.length() > maxFilenameChars)
                                maxFilenameChars = filename.length();
                            numExamples += 1;
                            if (addToClasses) {
                                mtx.lock();
                                classes.insert(className);
                                mtx.unlock();
                                addToClasses = false;
                            }
                        }
                    }
                }
            }

            for (auto it = classes.begin(); it != classes.end(); ++it) {
                if (it->length() > maxClassNameChars)
                    maxClassNameChars = it->length();
            }

            if (type == ExampleType::TRAIN)
                numTrainExamples += numExamples;
            else if (type == ExampleType::VALIDATE)
                numValidateExamples += numExamples;
            else if (type == ExampleType::TEST)
                numTestExamples += numExamples;
        }
    }
}

void ShardedRawDatasetCreator::loadExamples(WorkQueueUnordered<DataElement, DataElement>& workQueue) {
    assert(!sourceDirectories.empty());
    omp_set_num_threads(sourceDirectories.size());

    std::vector<string> sourceDirectoriesVector(sourceDirectories.begin(), sourceDirectories.end());
    uint32_t numSourceDirectories = sourceDirectoriesVector.size();

#pragma omp parallel for schedule(static, 1)
    for (uint32_t i = 0; i < numSourceDirectories; ++i) {
        string datasetDirectoryString = sourceDirectoriesVector[i];

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

            for (const directory_entry& classDirectory : directory_iterator(datasetDirectory)) {
                path classDirectoryPath = classDirectory.path();
                string filename = classDirectory.path().filename();
                bool filename_is_dot = filename == ".";
                bool filename_is_dot_dot = filename == "..";
                if (is_directory(classDirectoryPath) && !filename_is_dot && !filename_is_dot_dot) {
                    string className = classDirectoryPath.filename().native();
                    std::vector<path> examples;
                    if (classes.count(className) == 0)
                        continue;

                    for (const directory_entry& example : directory_iterator(classDirectoryPath)) {
                        if (is_regular_file(example.path())) {
                            path examplePath = example.path();
                            std::ifstream file(examplePath.native(), std::ios::binary | std::ios::ate);
                            uint64_t fileSizeInBytes = file.tellg();

                            // read file
                            file.seekg(0, std::ios::beg);
                            unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(fileSizeInBytes);
                            if (file.read((char*)buffer.get(), fileSizeInBytes)) {
                                DataElement rawDataElement;
                                rawDataElement.data = std::shared_ptr<uint8_t[]>(buffer.release(), std::default_delete<uint8_t[]>());
                                rawDataElement.numDataBytes = fileSizeInBytes;
                                rawDataElement.exampleType = type;
                                rawDataElement.className = className;
                                rawDataElement.fileName = examplePath.filename().native();

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

void ShardedRawDatasetCreator::writeDataToShard(WorkQueueUnordered<DataElement, DataElement>* workQueue,
                                                std::vector<std::shared_ptr<Shard>>* shards) {
    bool queueOpen = true;
    DataElement processedDataElement;
    string type;
    uint64_t destShard;
    while (true) {
        queueOpen = workQueue->pop(processedDataElement);
        if (!queueOpen)
            break;

        if (processedDataElement.numDataBytes == 0 || processedDataElement.data == nullptr)
            continue;
        if (processedDataElement.exampleType == ExampleType::TRAIN) {
            destShard = destShardTrain.fetch_add(1) % numOutputShards;
            type = "train";
        } else if (processedDataElement.exampleType == ExampleType::VALIDATE) {
            destShard = destShardValidate.fetch_add(1) % numOutputShards;
            type = "validate";
        } else {
            destShard = destShardTest.fetch_add(1) % numOutputShards;
            type = "test";
        }

        // printf("writing %s class %s to shard %ld\n", type.c_str(), processedDataElement.className.c_str(), destShard);
        (*shards)[destShard]->writeExample(processedDataElement.data.get(),
                                           processedDataElement.className,
                                           processedDataElement.fileName,
                                           processedDataElement.exampleType);
    }
    // printf("queue closed\n");
}
