#!/bin/bash

# This script is used to configure the meson build system for the RISC-V project
meson setup bmeson \
  --cross-file riscv_cross.txt \
  --buildtype=release \
  --unity=on --unity-size=2048
  
cd bmeson
# perform the build
meson compile


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
# as root run:
# echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="4348", ATTRS{idProduct}=="55e0", MODE="0666"' > /etc/udev/rules.d/50-wchisp.rules


./wchisp  flash risc_usb_test
