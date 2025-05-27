//
// Created by martin on 5/27/25.
//

// class B {
// public:
// B() { f3(10); }
//     // warning: pure virtual ‘virtual void B::f3(int)’ called from constructor
// virtual void f3(int x) = 0;
// };

class B {
    public:
    B(){ f1();}
    virtual void f1() {f3(10);}
    virtual void f3(int x) = 0;
};

int main() {
    B b; // error: cannot declare variable ‘b’ to be of abstract type ‘B’
    return 0;
}