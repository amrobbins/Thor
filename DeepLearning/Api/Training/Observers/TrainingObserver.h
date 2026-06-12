#pragma once

#include "DeepLearning/Api/Training/Events/TrainingEvent.h"

#include <memory>
#include <vector>

namespace Thor {

class TrainingObserver {
   public:
    virtual ~TrainingObserver() = default;
    virtual void onTrainingEvent(const TrainingEvent& event) = 0;
    virtual void flush() {}
    virtual void close() { flush(); }
};

class NullTrainingObserver : public TrainingObserver {
   public:
    void onTrainingEvent(const TrainingEvent& event) override { (void)event; }
    void flush() override {}
    void close() override {}
};

class CompositeTrainingObserver : public TrainingObserver {
   public:
    void addObserver(std::shared_ptr<TrainingObserver> observer) {
        if (observer) {
            observers.push_back(std::move(observer));
        }
    }

    [[nodiscard]] bool empty() const { return observers.empty(); }

    void onTrainingEvent(const TrainingEvent& event) override {
        for (const std::shared_ptr<TrainingObserver>& observer : observers) {
            observer->onTrainingEvent(event);
        }
    }

    void flush() override {
        for (const std::shared_ptr<TrainingObserver>& observer : observers) {
            if (observer) {
                observer->flush();
            }
        }
    }

    void close() override {
        for (const std::shared_ptr<TrainingObserver>& observer : observers) {
            if (observer) {
                observer->close();
            }
        }
    }

   private:
    std::vector<std::shared_ptr<TrainingObserver>> observers{};
};

}  // namespace Thor
