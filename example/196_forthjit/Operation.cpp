// no preamble
// implementation
#include "Operation.h"
Operation::Operation()
    : kind{OperationKind::Literal}, value{0}, primitive{Primitive::Add} {}
Operation::~Operation() {}