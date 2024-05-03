#ifndef OPERATOR_H
#define OPERATOR_H

// heaeder
 
#include <cstdint> 
class Operator  {
        public:
         Operator (uint8_t precedence = 0, uint8_t arguments = 0)       ;   
        const uint8_t& GetPrecedence () const      ;   
        void SetPrecedence (uint8_t precedence)       ;   
        const uint8_t& GetArguments () const      ;   
        void SetArguments (uint8_t arguments)       ;   
        public:
        uint8_t precedence;
        uint8_t arguments;
};

#endif /* !OPERATOR_H */