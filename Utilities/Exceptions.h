#pragma once

#include <stdexcept>
#include <string>

class NotYetImplementedException : public std::logic_error {
   public:
    NotYetImplementedException() : std::logic_error("Not yet implemented.") {}

    NotYetImplementedException(std::string message) : std::logic_error(message.c_str()) {}
};

class NotImplementedException : public std::logic_error {
   public:
    NotImplementedException() : std::logic_error("Not implemented.") {}

    NotImplementedException(std::string message) : std::logic_error(message.c_str()) {}
};