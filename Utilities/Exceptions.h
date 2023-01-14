#include <stdexcept>

class NotYetImplementedException : public std::logic_error {
   public:
    virtual char const* what() const _GLIBCXX_NOTHROW { return "Function not yet implemented."; }
};

class NotImplementedException : public std::logic_error {
   public:
    virtual char const* what() const _GLIBCXX_NOTHROW { return "Function not implemented."; }
};