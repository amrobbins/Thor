#pragma once
#include <memory>

struct CalledGetOnEmptyOptionalException : public std::exception {
    const char *what() const throw() { return "Tried to get the value of an empty optional."; }
};

template <typename T>
class Optional {
   public:
    Optional() { clear(); }

    Optional(const T &value) { set(value); }

    Optional(const Optional<T> &other) {
        // implemented using operator=
        *this = other;
    }

    bool isPresent() const { return present; }
    bool isEmpty() const { return !isPresent(); }
    void clear() {
        present = false;
        pValue.reset();
    }

    T &get() const {
        if (isEmpty()) {
            throw CalledGetOnEmptyOptionalException();
        }
        return *pValue;
    }

    operator T &() { return get(); }

    void set(const T &value) {
        present = true;
        this->pValue.reset(new T(value));
    }

    Optional<T> &operator=(const T &value) {
        set(value);
        return *this;
    }

    void set(const Optional<T> &other) {
        if (other.isPresent())
            set(other.get());
        else
            this->clear();
    }

    Optional<T> &operator=(const Optional<T> &other) {
        set(other);
        return *this;
    }

    static Optional<T> empty() { return Optional<T>(); }

   private:
    std::unique_ptr<T> pValue;
    bool present;
};
