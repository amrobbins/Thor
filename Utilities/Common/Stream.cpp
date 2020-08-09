#include "Utilities/Common/Stream.h"

#ifdef DEBUG_REF_COUNTS
atomic<int> Stream::streamsCreated(0);
atomic<int> Stream::streamsDestroyed(0);
atomic<int> Stream::cudnnHandlesCreated(0);
atomic<int> Stream::cudnnHandlesDestroyed(0);
Stream::RefCountChecker Stream::refCountChecker;
#endif
