enum class Primitive {
  Add,
  Sub,
  Mul,
  Dup,
  Drop,
  Swap,
  Dot,
  LessThan,
  GreaterThan,
  Equal,
  Fetch,
  Store
};
enum class OperationKind { Literal, Primitive, CallWord, If };
enum class ParseMode { Immediate, Definition };
enum class SequenceStop { End, Else, Then };