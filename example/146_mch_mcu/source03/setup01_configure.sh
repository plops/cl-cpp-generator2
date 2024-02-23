#export CC=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gcc
#export CXX=~/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-g++

export CC=~/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-gcc
export CXX=~/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-g++


mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
time ninja



# download wchisp if it doesn't already exist
if [ ! -f wchisp ]; then
    wget https://github.com/ch32-rs/wchisp/releases/download/nightly/wchisp-linux-x64.tar.gz
    tar xavf wchisp-linux-x64.tar.gz
fi


# to allow non-root users to access the USB device, create a udev rule
# if this file doesn't exist write a message for the user
if [ ! -f /etc/udev/rules.d/50-wchisp.rules ]; then
    echo "To allow non-root users to access the USB device, create a udev rule"
    echo "as root run:"
    echo "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"4348\", ATTRS{idProduct}==\"55e0\", MODE=\"0666\"' > /etc/udev/rules.d/50-wchisp.rules"
fi


./wchisp  flash risc_usb_test
