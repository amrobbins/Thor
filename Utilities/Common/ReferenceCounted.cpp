#include "Utilities/Common/ReferenceCounted.h"

atomic<long> ReferenceCounted::nextId(1L);

#ifdef DEBUG_REF_COUNTS
atomic<long> ReferenceCounted::objectsCreated(0L);
atomic<long> ReferenceCounted::objectsDestroyed(0L);
ReferenceCounted::RefCountChecker ReferenceCounted::refCountChecker;
#endif
