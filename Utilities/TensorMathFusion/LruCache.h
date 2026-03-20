#pragma once

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <class Key, class Value, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
class ThreadSafeLruCache {
   public:
    explicit ThreadSafeLruCache(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("ThreadSafeLruCache capacity must be > 0.");
        }
        map_.reserve(capacity_);
    }

    ThreadSafeLruCache(const ThreadSafeLruCache&) = delete;
    ThreadSafeLruCache& operator=(const ThreadSafeLruCache&) = delete;

    ThreadSafeLruCache(ThreadSafeLruCache&&) = delete;
    ThreadSafeLruCache& operator=(ThreadSafeLruCache&&) = delete;

    [[nodiscard]] std::size_t capacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return capacity_;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.size();
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.empty();
    }

    [[nodiscard]] bool contains(const Key& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.find(key) != map_.end();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.clear();
        entries_.clear();
    }

    void setCapacity(std::size_t new_capacity) {
        if (new_capacity == 0) {
            throw std::invalid_argument("ThreadSafeLruCache capacity must be > 0.");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = new_capacity;
        evictIfNeededLocked();
        map_.reserve(capacity_);
    }

    template <class K, class V>
    void put(K&& key, V&& value) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second->value = std::forward<V>(value);
            touchLocked(it->second);
            return;
        }

        entries_.push_front(Entry{std::forward<K>(key), std::forward<V>(value)});
        map_[entries_.front().key] = entries_.begin();
        evictIfNeededLocked();
    }

    [[nodiscard]] std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }

        touchLocked(it->second);
        return it->second->value;
    }

    bool tryGet(const Key& key, Value& out_value) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }

        touchLocked(it->second);
        out_value = it->second->value;
        return true;
    }

    [[nodiscard]] bool erase(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }

        entries_.erase(it->second);
        map_.erase(it);
        return true;
    }

   private:
    struct Entry {
        Key key;
        Value value;
    };

    using List = std::list<Entry>;
    using ListIt = typename List::iterator;
    using Map = std::unordered_map<Key, ListIt, Hash, KeyEqual>;

    void touchLocked(ListIt it) {
        if (it != entries_.begin()) {
            entries_.splice(entries_.begin(), entries_, it);
        }
    }

    void evictIfNeededLocked() {
        while (map_.size() > capacity_) {
            auto last = std::prev(entries_.end());
            map_.erase(last->key);
            entries_.pop_back();
        }
    }

    mutable std::mutex mutex_;
    std::size_t capacity_;
    List entries_;  // most recently used at front
    Map map_;
};
