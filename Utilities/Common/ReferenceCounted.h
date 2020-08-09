#pragma once

#include <assert.h>
#include <stdio.h>
#include <atomic>

using std::atomic;

/**
 * Note: if you want to create a class that derives from a class that derives ReferenceCounted,
 *       you will have to be able to figure out which is the most derived destructor and only
 *       it will call removeReference(). Then the intermediate classes will have to be able to
 *       ask if they should destroy without decrementing the reference count. This support will
 *       need to be added if this pattern is desired.
 *
 *       Since that seems a bit messy, it is not really recommended to do that, unless you really have to.
 */

#define DEBUG_REF_COUNTS

class ReferenceCounted {
   public:
    ReferenceCounted() {
        initialized = false;
        referenceCount = nullptr;
    }

    ReferenceCounted(const ReferenceCounted &other) {
        initialized = false;
        referenceCount = nullptr;

        *this = other;  // implemented using operator=
    }

    ReferenceCounted &operator=(const ReferenceCounted &other) {
        // Do not reorder the increment/decrement of refCount here or object may be destroyed prematurely
        if (other.initialized) {
            // other is initialized
            other.referenceCount->fetch_add(1);
            if (initialized) {
                // this stream was previously initialized
                removeReference();
            }
            initialized = true;
            referenceCount = other.referenceCount;

            return *this;
        } else {
            // other stream is not initialized
            if (initialized) {
                // this stream was previously initialized
                removeReference();
            }
            initialized = false;
            referenceCount = nullptr;
            return *this;
        }
    }

    void initialize() {
        referenceCount = new atomic<int>(1);
        initialized = true;

#ifdef DEBUG_REF_COUNTS
        objectsCreated.fetch_add(1);
#endif
    }

    virtual ~ReferenceCounted() {}

    bool uninitialized() const { return !initialized; }

   protected:
    // Returns true when the derived class should destroy itself.
    bool removeReference() {
        if (!initialized) {
            assert(referenceCount == nullptr);
            return false;
        }

        int refCountBeforeDecrement = referenceCount->fetch_sub(1);

        if (refCountBeforeDecrement == 1) {
            delete referenceCount;
            referenceCount = nullptr;
            initialized = false;

#ifdef DEBUG_REF_COUNTS
            objectsDestroyed.fetch_add(1);
#endif

            // Tell the derived class to destroy itself
            return true;
        }

        return false;
    }

   private:
    bool initialized;
    atomic<int> *referenceCount;

#ifdef DEBUG_REF_COUNTS
    static atomic<long> objectsCreated;
    static atomic<long> objectsDestroyed;

    class RefCountChecker {
       public:
        virtual ~RefCountChecker() {
            printf("reference counted objects created %ld reference counted objects destroyed %ld\n",
                   objectsCreated.fetch_add(0),
                   objectsDestroyed.fetch_add(0));
            fflush(stdout);
        }
    };
    static RefCountChecker refCountChecker;
#endif
};
