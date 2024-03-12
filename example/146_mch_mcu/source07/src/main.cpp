#include <cstdint>
#include <iostream>
#include <ostream>

class A
{
public:
    struct Ctl
    {
        union
        {
            uint8_t reg;
            struct
            {
                uint8_t enable : 1;
                uint8_t start_bit_align : 1;
                uint8_t bit_align_finished : 1;
                uint8_t bit3 : 1;
                uint8_t bit4 : 1;
                uint8_t bit5 : 1;
                uint8_t bit6 : 1;
                uint8_t bit7 : 1;
            };
        };
        Ctl &operator=(uint8_t val)
        {
            reg = val;
            return *this;
        }
        std::ostream &print(std::ostream &os) const
        {
            os << "enable: " << static_cast<int>(enable) << std::endl;
            os << "start_bit_align: " << static_cast<int>(start_bit_align) << std::endl;
            os << "bit_align_finished: " << static_cast<int>(bit_align_finished) << std::endl;
            return os;
        }
    } ctl;

    struct Status
    {
        union
        {
            uint8_t reg;
            struct
            {
                uint8_t enabled : 1;
                uint8_t bit_align_running : 1;
                uint8_t bit_align_error : 1;
                uint8_t reserved : 5;
            };
        };
        Status &operator=(uint8_t val)
        {
            reg = val;
            return *this;
        }
        std::ostream &print(std::ostream &os) const
        {
            os << "enabled: " << static_cast<int>(enabled) << std::endl;
            os << "bit_align_running: " << static_cast<int>(bit_align_running) << std::endl;
            os << "bit_align_error: " << static_cast<int>(bit_align_error) << std::endl;
            return os;
        }
    } status;
};

int main(int argc, char **argv)
{
    A a;
    a.ctl = 0xaf;
    a.status = 0x07;
    // print all bits

    a.ctl.print(std::cout) << std::endl;
    a.status.print(std::cout) << std::endl;

    /*
    source07/src $ ./a.out
    enable: 1
    start_bit_align: 1
    bit_align_finished: 1

    enabled: 1
    bit_align_running: 1
    bit_align_error: 1
    */
    return 0;
}