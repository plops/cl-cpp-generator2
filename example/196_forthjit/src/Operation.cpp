// no preamble
// implementation
#include "Operation.h"
Operation::Operation()
    : kind{OperationKind::Literal}, value{0}, primitive{Primitive::Add} {}
Operation::~Operation() {}
Operation Operation::literal(int value) {
  auto op{Operation{}};
  op.kind = OperationKind::Literal;
  op.value = value;
  return op;
}
Operation Operation::primitive_op(Primitive primitive) {
  auto op{Operation{}};
  op.kind = OperationKind::Primitive;
  op.primitive = primitive;
  return op;
}
Operation Operation::call_word(int word_index) {
  auto op{Operation{}};
  op.kind = OperationKind::CallWord;
  op.value = word_index;
  return op;
}
Operation Operation::if_op(std::vector<Operation> true_branch,
                           std::vector<Operation> false_branch) {
  auto op{Operation{}};
  op.kind = OperationKind::If;
  op.true_branch = std::move(true_branch);
  op.false_branch = std::move(false_branch);
  return op;
}