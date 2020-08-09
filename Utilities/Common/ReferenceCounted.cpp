#include "Utilities/Common/ReferenceCounted.h"

#ifdef DEBUG_REF_COUNTS
atomic<int> ReferenceCounted::objectsCreated(0);
atomic<int> ReferenceCounted::objectsDestroyed(0);
ReferenceCounted::RefCountChecker ReferenceCounted::refCountChecker;
#endif
