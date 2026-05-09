#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace ThorImplementation::ThorError {

[[noreturn]] inline void throwFailedCheck(const char *condition,
                                          const char *file,
                                          int line,
                                          const char *function) {
    std::ostringstream message;
    message << "Thor check failed: " << condition << " at " << file << ':' << line << " in " << function << "().";
    throw std::logic_error(message.str());
}

[[noreturn]] inline void throwUnreachable(const char *file, int line, const char *function) {
    std::ostringstream message;
    message << "Thor reached code that should be unreachable at " << file << ':' << line << " in " << function << "().";
    throw std::logic_error(message.str());
}

inline void check(bool condition, const char *conditionText, const char *file, int line, const char *function) {
    if (!condition)
        throwFailedCheck(conditionText, file, line, function);
}

}  // namespace ThorImplementation::ThorError

#define THOR_THROW_IF_FALSE(...) \
    ::ThorImplementation::ThorError::check(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__, __func__)

#define THOR_UNREACHABLE() ::ThorImplementation::ThorError::throwUnreachable(__FILE__, __LINE__, __func__)
