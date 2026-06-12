#pragma once

#include "DeepLearning/Api/Training/Events/TrainingEvent.h"

#include <memory>
#include <vector>

namespace Thor {

class TrainingObserver {
   public:
    virtual ~TrainingObserver() = default;
    virtual void onTrainingEvent(const TrainingEvent& event) = 0;
};

class NullTrainingObserver : public TrainingObserver {
   public:
    void onTrainingEvent(const TrainingEvent& event) override { (void)event; }
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

   private:
    std::vector<std::shared_ptr<TrainingObserver>> observers{};
};

}  // namespace Thor
