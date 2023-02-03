#ifndef USBERROR_H
#define USBERROR_H

#include <exception>
#include <stdexcept>
#include <libusb-1.0/libusb.h>


void check (int err)    ;  
class UsbError : public std::runtime_error {
        public:
         UsbError (int err_code)     ;  
        int code () const    ;  
        private:
        int _code;
};

#endif /* !USBERROR_H */