#ifndef ERROR_H
#define ERROR_H

#include <stdexcept>

class Error : public std::runtime_error {
        public:
         Error ()     ;  
        int code () const    ;  
        private:
        int _code;
};

#endif /* !ERROR_H */