#export CC=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gcc
#export CXX=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-g++

export CC=~/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-gcc
export CXX=~/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-g++


mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
time ninja
# turns out i can flash elf files. conversion to hex is not necessary
# https://youtu.be/L9Wrv7nW-S8?t=626
#~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-objcopy -O ihex risc_test risc_test.hex 
#~/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-objcopy 
 
wget https://github.com/ch32-rs/wchisp/releases/download/nightly/wchisp-linux-x64.tar.gz
tar xavf wchisp-linux-x64.tar.gz

# as root run:
# echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="4348", ATTRS{idProduct}=="55e0", MODE="0666"' > /etc/udev/rules.d/50-wchisp.rules


./wchisp  flash risc_test
