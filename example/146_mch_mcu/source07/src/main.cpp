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
    std::cout << "bit0: " << static_cast<int>(a.ctl.bit0) << std::endl;
    std::cout << "bit1: " << static_cast<int>(a.ctl.bit1) << std::endl;
    std::cout << "bit2: " << static_cast<int>(a.ctl.bit2) << std::endl;
    std::cout << "bit3: " << static_cast<int>(a.ctl.bit3) << std::endl;
    std::cout << "bit4: " << static_cast<int>(a.ctl.bit4) << std::endl;
    std::cout << "bit5: " << static_cast<int>(a.ctl.bit5) << std::endl;
    std::cout << "bit6: " << static_cast<int>(a.ctl.bit6) << std::endl;
    std::cout << "bit7: " << static_cast<int>(a.ctl.bit7) << std::endl;
    /* /source07/src $ ./a.out 
bit0: 1
bit1: 1
bit2: 1
bit3: 1
bit4: 0
bit5: 1
bit6: 0
bit7: 1
    */
    return 0;
}