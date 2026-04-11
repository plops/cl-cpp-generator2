// no preamble
// implementation
#include "Operation.h"
Operation::Operation()
    : kind{OperationKind::Literal}, value{0}, primitive{Primitive::Add} {}
Operation::~Operation() {}
Operation Operation::literal(int value) {
  auto op{Operation{.kind = OperationKind::Literal, .value = value}};
  return op;
}