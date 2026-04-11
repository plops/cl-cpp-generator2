// no preamble
#include "Operation.h"
#include <iostream>
public:
Operation();
OperationKind kind{OperationKind::Literal};
int value{0};
Primitive primitive{Primitive::Add};
std::vector<Operation> true_branch;
std::vector<Operation> false_branch;
void Operation::Operation(OperationKind k, int v) : kind{k}, value{v} { {}; }
void Operation::Operation(OperationKind k, Primitive p)
    : kind{k}, primitive{p} {
  {};
}
void Operation::Operation(OperationKind k, std::vector<Operation> t_,
                          std::vector<Operation> f_)
    : kind{k}, true_branch{t_}, false_branch{f_} {
  {};
}

Operation Operation::Literal(int v) {
  auto op{{OperationKind::Literal, v}};
  return op;
}

Operation Operation::Prim(Primitive p) {
  auto op{{OperationKind::Primitive, p}};
  return op;
}

Operation Operation::CallWord(int v) {
  auto op{{OperationKind::CallWord, v}};
  return op;
}

Operation Operation::If(std::vector<Operation> t_, std::vector<Operation> f_) {
  auto op{{OperationKind::If, t_, f_}};
  return op;
}
