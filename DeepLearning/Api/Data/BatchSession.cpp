#include "DeepLearning/Api/Data/BatchSession.h"

#include <stdexcept>

namespace Thor {

BatchLease::BatchLease(std::shared_ptr<BatchSession> session, ExampleType exampleType, Batch batch)
    : session(std::move(session)), exampleType(exampleType), batch(std::move(batch)) {
    if (this->session == nullptr) {
        throw std::runtime_error("BatchLease requires a BatchSession.");
    }
}

BatchLease::~BatchLease() { reset(); }

BatchLease::BatchLease(BatchLease &&other) noexcept
    : session(std::move(other.session)), exampleType(other.exampleType), batch(std::move(other.batch)) {}

BatchLease &BatchLease::operator=(BatchLease &&other) noexcept {
    if (this != &other) {
        reset();
        session = std::move(other.session);
        exampleType = other.exampleType;
        batch = std::move(other.batch);
    }
    return *this;
}

Batch &BatchLease::get() {
    if (session == nullptr) {
        throw std::runtime_error("BatchLease is empty.");
    }
    return batch;
}

const Batch &BatchLease::get() const {
    if (session == nullptr) {
        throw std::runtime_error("BatchLease is empty.");
    }
    return batch;
}

Batch BatchLease::release() {
    if (session == nullptr) {
        throw std::runtime_error("BatchLease is empty.");
    }
    session.reset();
    return std::move(batch);
}

void BatchLease::reset() noexcept {
    if (session != nullptr) {
        try {
            session->returnBatchBuffers(exampleType, std::move(batch));
        } catch (...) {
            // Destruction must not terminate a run while unwinding another exception.
        }
        session.reset();
    }
}

}  // namespace Thor
