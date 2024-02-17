export CC=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gcc
export CXX=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-g++
mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
