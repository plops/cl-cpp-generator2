# Cheap Risc-V microcontroller

## Introduction

- i want to write some code for MCH microcontrollers with risc-v


## Install dependencies

- download the sdk and example programs:

```
mkdir ~/risc
cd risc
wget http://file.mounriver.com/tools/MRS_Toolchain_Linux_x64_V1.90.tar.xz
# 308MB
tar xaf MRS_Toolchain_Linux_x64_V1.90.tar.xz 
- contains GCC12 (binaries), OpenOCD, datasheet for MCU

cd ~/src
git clone https://github.com/openwch/ch592 # 87MB
 
 
 
git clone https://github.com/WeActStudio/WeActStudio.WCH-BLE-Core # 178MB
- contains board schematic, datasheet for MCU
- SDK and USB flash tool (but only for windows)
- Example code
- I think this is a good place to start: Examples/CH592/ble/broadcaster/User/main.c
```


## Aliexpress order

```
CH32V003 industrial grade 32-bit general-purpose RISC-V MCU 10Pcs/lot
CH32V003J4M6 SOP8
CHF1.91x1


WCH LinkE Online Download Debugger Support WCH RISC-V Architecture MCU/SWD Interface ARM Chip 1 Serial Port to USB Channel
1Pcs WCH-LinkE
CHF4.11x1


WeAct CH592F CH592 RISC-V Core BLE5.4 Wireless MCU WCH Core Board Demo Board
1PCS
CHF2.11x2
```

## Gpio Example


## Configure Visual Studio Code

- install cmake extension in visual studio code and specify the risc-v
  compiler as a cmake kit by adding the following settings to
  ~/.local/share/CMakeTools/cmake-tools-kits.json

```
  {
    "name": "GCC 12.2 risc-v",
    "compilers": {
      "C": "/home/martin/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gcc",
      "CXX": "/home/martin/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-g++"
    },
    "isTrusted": true
  }
```

- go to cmake pane, configure the project, press build

## Configure CLion

- go to Settings->Build, Execution, Deployment->Toolchains
- create a new toolchain by pressing the Plus "+" icon in the top left
- name the toolchain. i named it Riscv
- specify the three entries for C compiler, C++ Compiler and Debugger with absolute paths. in my configuration it looks like this

```
/home/martin/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gcc
/home/martin/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-g++
/home/martin/risc/RISC-V_Embedded_GCC12/bin/riscv-none-elf-gdb
```

- in  Settings->Build, Execution, Deployment->CMake create a Release configuration that uses this toolchain and press apply
- press build
