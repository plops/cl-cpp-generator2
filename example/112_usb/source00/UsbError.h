#ifndef USBERROR_H
#define USBERROR_H

#include <stdexcept>


void check (int err)    ;  
class UsbError : public std::runtime_error {
        public:
         UsbError ()     ;  
        int code () const    ;  
        private:
        int _code;
};

#endif /* !USBERROR_H */