#ifndef SYMBOL_H
#define SYMBOL_H

// heaeder
 
#include <cstdint>
#include <string> 
#include "Operator.h" 
enum class Type : uint8_t {
        Unknown, Literal_Numeric, Operator 
};
class Symbol  {
        public:
         Symbol (std::string symbol = "", Type type = Type::Unknown, Operator op = {})       ;   
        const std::string& GetSymbol () const      ;   
        void SetSymbol (std::string symbol)       ;   
        const Type& GetType () const      ;   
        void SetType (Type type)       ;   
        const Operator& GetOp () const      ;   
        void SetOp (Operator op)       ;   
        public:
        std::string symbol;
        Type type;
        Operator op;
};

#endif /* !SYMBOL_H */