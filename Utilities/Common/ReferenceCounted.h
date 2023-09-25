#pragma once

#include <assert.h>
#include <stdio.h>
#include <atomic>
#include <mutex>

/**
 * Note: if you want to create a class that derives from a class that derives ReferenceCounted,
 *       you will have to be able to figure out which is the most derived destructor and only
 *       it will call removeReference(). Then the intermediate classes will have to be able to
 *       ask if they should destroy without decrementing the reference count. This support will
 *       need to be added if this pattern is desired.
 *
 *       Since that seems a bit messy, it is not really recommended to do that, unless you really have to.
 */

/**
 * Note: This is the thread related guarantee that ReferenceCounted gives:
 *
 * Say that Tensor is a reference counted type.
 * Initial:
 *  Tensor tensor(...) is an initialized tensor.
 *  The constructor Tensor() creates an uninitialized tensor.
 *
 *   thread 0                               thread 1
 *   --------                               --------
 *   tensorToConnect = tensor;              tensor = Tensor();
 *
 * So thread 0 is trying to refer to tensor via tensorToConnect,
 * but thread 1 is uninitializing tensor in parallel.
 * In this case the guarantee is that one of 2 outcomes will occur:
 *
 * 1. The object initially referred to by tensorToConnect will be destroyed.
 *    tensorToConnect will refer to the object originally referred to by tensor.
 *    tensor will end up referring to no object (reference count of 0).
 *    So the object originally referred to by tensor was not destroyed, it is being referred to by tensorToConnect.
 *    So tensorToConnect will refer to a tensor that has a referenceCount of 1.
 * 2. Both tensorToConnect and tensor will end up referring to no object (reference count of 0),
 *    the tensor will be destroyed when its reference count goes to 0 in thread 1.
 *
 *
 * Also consider the following case:
 * Initial:
 *  tensor1 = Tensor(...);
 *  tensor2 = Tensor(...);
 *  tensor3 = Tensor(...);
 *
 * And those are the only references to those tensors;
 *
 *   thread 0                                thread 1
 *   --------                                --------
 *   tensor1 = tensor2;                      tensor2 = tensor3;
 *
 * In this case the original object referenced by tensor1 will be destroyed.
 * Then tensor1 is assigned to something.
 * If the assignment of class members from tensor3 to tensor2 is not atomic,
 * then tensor1's member objects may end up being filled with something that is
 * part tensor2 and part tensor3. ReferenceCounted does not handle that, it is up
 * to the class that derives from ReferenceCounted to handle that if necessary.
 * In terms of the referenceCounts, the possible cases are:
 *
 * 1. The object originally referred to by tensor1 is destroyed.
 *    The object originally referred to by tensor2 is destroyed.
 *    tensor1, tensor2 and tensor3 refer to an object with a reference count of 3.
 * 2. The object originally referred to by tensor1 is destroyed.
 *    tensor1 refers to the object originally referred to by tensor2,
 *    so tensor1 points to an object with a reference count of 1.
 *    tensor2 and tensor3 refer to an object with a reference count of 2.
 *
 * ReferenceCounted provides lockSelfAndOther() and unlockSelfAndOther() which allows any two
 * ReferenceCounted objects to be locked at the same time without the possibility of deadlock
 * (via strict ordering based on unique id), this allows derived classes to lock
 * when changing the object that they refer to, if the derived class has a need to do that.
 *
 */
// #define DEBUG_REF_COUNTS

class ReferenceCounted {
   public:
    ReferenceCounted() : referenceCount(nullptr), id(nextId.fetch_add(1)) {}

    ReferenceCounted(const ReferenceCounted &other) : referenceCount(nullptr), id(nextId.fetch_add(1)) {
        *this = other;  // implemented using operator=
    }

    uint32_t getReferenceCountedId() const { return id; }

    ReferenceCounted &operator=(const ReferenceCounted &other) {
        std::atomic<uint32_t> *otherReferenceCount = other.referenceCount;

        // If they already refer to the same instance, or they are both uninitialized, then do nothing.
        if (referenceCount == otherReferenceCount)
            return *this;

        if (otherReferenceCount != nullptr) {
            // other is initialized
            otherReferenceCount->fetch_add(1);
            if (referenceCount != nullptr) {
                // this one was previously initialized
                bool shouldDestroy = removeReference();
                if (shouldDestroy)
                    this->destroy();
            }
            referenceCount = otherReferenceCount;

            return *this;
        } else {
            // other is not initialized
            if (referenceCount != nullptr) {
                // this one was previously initialized
                bool shouldDestroy = removeReference();
                if (shouldDestroy)
                    this->destroy();
            }
            referenceCount = nullptr;
            return *this;
        }
    }

    // Lock 2 ReferenceCounted's without possibility of deadlock
    void lockSelfAndOther(ReferenceCounted &other) {
        if (id < other.id) {
            mtx.lock();
            other.mtx.lock();
        } else {
            other.mtx.lock();
            mtx.lock();
        }
    }

    void unlockSelfAndOther(ReferenceCounted &other) {
        if (id < other.id) {
            other.mtx.unlock();
            mtx.unlock();
        } else {
            mtx.unlock();
            other.mtx.unlock();
        }
    }

    uint32_t getReferenceCount() { return referenceCount.load()->load(); }

    void initialize() {
        referenceCount = new std::atomic<uint32_t>(1);

#ifdef DEBUG_REF_COUNTS
        objectsCreated.fetch_add(1);
#endif
    }

    virtual ~ReferenceCounted() = default;

    bool uninitialized() const { return referenceCount == nullptr; }
    bool initialized() const { return !uninitialized(); }

    virtual void destroy() = 0;

   protected:
    // Returns true when the derived class should destroy itself.
    bool removeReference() {
        if (uninitialized()) {
            return false;
        }

        uint32_t refCountBeforeDecrement = referenceCount.load()->fetch_sub(1);

        if (refCountBeforeDecrement == 1) {
            delete referenceCount.load();
            referenceCount = nullptr;

#ifdef DEBUG_REF_COUNTS
            objectsDestroyed.fetch_add(1);
#endif

            // Tell the derived class to destroy itself
            return true;
        }

        return false;
    }

   private:
    std::atomic<std::atomic<uint32_t> *> referenceCount;

    std::recursive_mutex mtx;

    const long id;
    static std::atomic<long> nextId;

#ifdef DEBUG_REF_COUNTS
    static std::atomic<long> objectsCreated;
    static std::atomic<long> objectsDestroyed;

    class RefCountChecker {
       public:
        virtual ~RefCountChecker() {
            printf("reference counted objects created %ld reference counted objects destroyed %ld\n",
                   objectsCreated.fetch_add(0),
                   objectsDestroyed.fetch_add(0));
            fflush(stdout);
            assert(objectsCreated.fetch_add(0) == objectsDestroyed.fetch_add(0));
        }
    };
    static RefCountChecker refCountChecker;
#endif
};
