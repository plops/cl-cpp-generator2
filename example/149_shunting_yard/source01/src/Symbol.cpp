// no preamble

// implementation

#include "Symbol.h"
Symbol::Symbol(std::string symbol, Type type, Operator op)
    : symbol{symbol}, type{type}, op{op} {}
const std::string &Symbol::GetSymbol() const { return symbol; }
void Symbol::SetSymbol(std::string symbol) { this->symbol = symbol; }
const Type &Symbol::GetType() const { return type; }
void Symbol::SetType(Type type) { this->type = type; }
const Operator &Symbol::GetOp() const { return op; }
void Symbol::SetOp(Operator op) { this->op = op; }