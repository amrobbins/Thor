#include "Utilities/Common/ReferenceCounted.h"

#ifdef DEBUG_REF_COUNTS
atomic<long> ReferenceCounted::objectsCreated(0L);
atomic<long> ReferenceCounted::objectsDestroyed(0L);
atomic<long> ReferenceCounted::nextId(1L);
ReferenceCounted::RefCountChecker ReferenceCounted::refCountChecker;
#endif
