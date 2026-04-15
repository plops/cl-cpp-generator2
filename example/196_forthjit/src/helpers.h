#pragma once
enum class Error : int {
  kOk = 0,
  Unknown_Word = 1,
  Stack_Error = 2,
  Compile_Error = 3,
  Dictionary_Full = 4,
  Invalid_Fuel = 5
};

inline const char *error_name(Error error) {
  switch (error) {
  case Error::Unknown_Word: {
    return "Unknown_Word";
    break;
  };
  case Error::Stack_Error: {
    return "Stack_Error";
    break;
  };
  case Error::Compile_Error: {
    return "Compile_Error";
    break;
  };
  case Error::Dictionary_Full: {
    return "Dictionary_Full";
    break;
  };
  case Error::Invalid_Fuel: {
    return "Invalid_Fuel";
    break;
  };
  case Error::kOk: {
    return "kOk";
    break;
  };
  }
  return "Error";
}

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