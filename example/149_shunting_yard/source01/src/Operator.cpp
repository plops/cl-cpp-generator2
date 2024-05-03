// no preamble

// implementation

#include "Operator.h"
Operator::Operator(uint8_t precedence, uint8_t arguments)
    : precedence{precedence}, arguments{arguments} {}
const uint8_t &Operator::GetPrecedence() const { return precedence; }
void Operator::SetPrecedence(uint8_t precedence) {
  this->precedence = precedence;
}
const uint8_t &Operator::GetArguments() const { return arguments; }
void Operator::SetArguments(uint8_t arguments) { this->arguments = arguments; }