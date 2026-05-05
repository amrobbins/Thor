#pragma once

#include <exception>
#include <optional>
#include <utility>

struct CalledGetOnEmptyOptionalException : public std::exception {
    const char* what() const throw() override { return "Tried to get the value of an empty optional."; }
};

template <typename T>
class Optional {
   public:
    Optional() = default;
    Optional(const T& value) : value(value) {}
    Optional(T&& value) : value(std::move(value)) {}

    Optional(const Optional<T>& other) = default;
    Optional(Optional<T>&& other) noexcept = default;
    Optional<T>& operator=(const Optional<T>& other) = default;
    Optional<T>& operator=(Optional<T>&& other) noexcept = default;

    bool isPresent() const { return value.has_value(); }
    bool isEmpty() const { return !isPresent(); }

    void clear() { value.reset(); }

    T& get() {
        if (isEmpty()) {
            throw CalledGetOnEmptyOptionalException();
        }
        return *value;
    }

    const T& get() const {
        if (isEmpty()) {
            throw CalledGetOnEmptyOptionalException();
        }
        return *value;
    }

    operator T&() { return get(); }
    operator const T&() const { return get(); }

    void set(const T& newValue) { value = newValue; }
    void set(T&& newValue) { value = std::move(newValue); }

    Optional<T>& operator=(const T& newValue) {
        set(newValue);
        return *this;
    }

    Optional<T>& operator=(T&& newValue) {
        set(std::move(newValue));
        return *this;
    }

    void set(const Optional<T>& other) { value = other.value; }
    void set(Optional<T>&& other) { value = std::move(other.value); }

    static Optional<T> empty() { return Optional<T>(); }

   private:
    std::optional<T> value;
};
