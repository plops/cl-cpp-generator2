//
// Created by martin on 5/27/25.
//

class B {
public:
B() { f3(10); }
    // warning: pure virtual ‘virtual void B::f3(int)’ called from constructor
virtual void f3(int x) = 0;
};