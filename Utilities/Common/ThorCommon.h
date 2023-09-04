#pragma once

#ifdef THOR_DEBUG
#define DEBUG_FLUSH_STDOUT() fflush(stdout)
#else
#define DEBUG_FLUSH_STDOUT() ((void)0)
#endif