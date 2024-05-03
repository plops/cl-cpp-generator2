#ifndef OPERATOR_H
#define OPERATOR_H

// heaeder
 
class Operator  {
        public:
        explicit  Operator (uint8_t precedence, uint8_t arguments)       ;   
        const uint8_t& GetPrecedence () const      ;   
        void SetPrecedence (uint8_t precedence)       ;   
        const uint8_t& GetArguments () const      ;   
        void SetArguments (uint8_t arguments)       ;   
        private:
        uint8_t precedence;
        uint8_t arguments;
};

#endif /* !OPERATOR_H */