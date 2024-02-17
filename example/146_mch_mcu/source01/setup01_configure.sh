export CC=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gcc
export CXX=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-g++

mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
time ninja
# https://youtu.be/L9Wrv7nW-S8?t=626
~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-objcopy -O ihex risc_test risc_test.hex 
