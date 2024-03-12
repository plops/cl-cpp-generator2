#include <cstdint>
#include <iostream>

class A {
    public:
    union {
        uint8_t reg;
        struct {
            uint8_t bit0 : 1;
            uint8_t bit1 : 1;
            uint8_t bit2 : 1;
            uint8_t bit3 : 1;
            uint8_t bit4 : 1;
            uint8_t bit5 : 1;
            uint8_t bit6 : 1;
            uint8_t bit7 : 1;
        };
    } ctl;
};

int main(int argc, char**argv)
{
    A a;
    a.ctl.reg = 0xaf;
    // print all bits
    std::cout << "bit0: " << a.ctl.bit0 << std::endl;
    std::cout << "bit1: " << a.ctl.bit1 << std::endl;
    std::cout << "bit2: " << a.ctl.bit2 << std::endl;
    std::cout << "bit3: " << a.ctl.bit3 << std::endl;
    std::cout << "bit4: " << a.ctl.bit4 << std::endl;
    std::cout << "bit5: " << a.ctl.bit5 << std::endl;
    std::cout << "bit6: " << a.ctl.bit6 << std::endl;
    std::cout << "bit7: " << a.ctl.bit7 << std::endl;
    
    return 0;
}