#pragma once

#include "DeepLearning/Api/Training/Events/TrainingStatsEvent.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

#include <memory>
#include <string>
#include <utility>

namespace Thor {

class TrainingStatsSink {
   public:
    virtual ~TrainingStatsSink() = default;
    virtual void onStatsEvent(const TrainingStatsEvent& event) = 0;
    virtual void flush() {}
    virtual void close() { flush(); }
};

class NullTrainingStatsSink : public TrainingStatsSink {
   public:
    void onStatsEvent(const TrainingStatsEvent& event) override { (void)event; }
    void flush() override {}
    void close() override {}
};

class TrainingStatsSinkObserver : public TrainingObserver {
   public:
    explicit TrainingStatsSinkObserver(std::shared_ptr<TrainingStatsSink> sink,
                                       std::string runName = {},
                                       bool flushSinkOnFlush = true)
        : sink(std::move(sink)), runName(std::move(runName)), flushSinkOnFlush(flushSinkOnFlush) {}

    void onTrainingEvent(const TrainingEvent& event) override {
        if (sink == nullptr) {
            return;
        }
        sink->onStatsEvent(TrainingStatsEvent::fromTrainingEvent(event, runName));
    }

    void flush() override {
        if (sink != nullptr && flushSinkOnFlush) {
            sink->flush();
        }
    }

    void close() override {
        if (sink != nullptr) {
            sink->close();
        }
    }

   private:
    std::shared_ptr<TrainingStatsSink> sink{};
    std::string runName{};
    bool flushSinkOnFlush = true;
};

}  // namespace Thor
