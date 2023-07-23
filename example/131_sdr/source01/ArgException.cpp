// no preamble



#include "ArgException.h"

ArgException::ArgException(std::string msg) : msg_(msg) {
}

const char *ArgException::what() const noexcept {
    return msg_.c_str();
} 
 
