// no preamble
// implementation
#include "MyClass.h"
MyClass::MyClass() {}
MyClass::~MyClass() {}
const int &MyClass::GetValue() { return value; }
void MyClass::SetValue(int value) { this->value = value; }