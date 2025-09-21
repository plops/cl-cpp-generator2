// no preamble
// implementation
#include "MyClass.h"
MyClass::MyClass() {}
MyClass::~MyClass() {}
const int &MyClass::getValue() { return value; }
void MyClass::setValue(int value) { this->value = value; }