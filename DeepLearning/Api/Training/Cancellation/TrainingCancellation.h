#pragma once

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>

namespace Thor {

class TrainingCancelled : public std::runtime_error {
   public:
    explicit TrainingCancelled(const std::string& message = "Training cancelled.") : std::runtime_error(message) {}
};

class TrainingInterrupted : public std::runtime_error {
   public:
    explicit TrainingInterrupted(const std::string& message = "Training interrupted.") : std::runtime_error(message) {}
};

class TrainingCancellationToken {
   public:
    TrainingCancellationToken() : requested(std::make_shared<std::atomic_bool>(false)) {}

    [[nodiscard]] bool isCancellationRequested() const { return requested != nullptr && requested->load(std::memory_order_acquire); }

    void throwIfCancellationRequested(const std::string& message = "Training cancelled.") const {
        if (isCancellationRequested()) {
            throw TrainingCancelled(message);
        }
    }

   private:
    explicit TrainingCancellationToken(std::shared_ptr<std::atomic_bool> requested) : requested(std::move(requested)) {}

    std::shared_ptr<std::atomic_bool> requested;

    friend class TrainingCancellationSource;
};

class TrainingCancellationSource {
   public:
    TrainingCancellationSource() : requested(std::make_shared<std::atomic_bool>(false)) {}

    [[nodiscard]] TrainingCancellationToken token() const { return TrainingCancellationToken(requested); }

    void requestCancellation() const { requested->store(true, std::memory_order_release); }

    [[nodiscard]] bool isCancellationRequested() const { return requested->load(std::memory_order_acquire); }

   private:
    std::shared_ptr<std::atomic_bool> requested;
};

}  // namespace Thor
