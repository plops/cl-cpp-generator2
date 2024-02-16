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
