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

## Use OpenOCD to program the device

I am reading this article: https://robbert-groen.medium.com/getting-started-with-the-gd32vf103-risc-v-microcontroller-22cb34718b0d


### The programmer

- dmesg output when plugging in:
```
[ 9065.746131] usb 3-1: new full-speed USB device number 3 using xhci_hcd
[ 9065.911432] usb 3-1: New USB device found, idVendor=1a86, idProduct=8012, bcdDevice= 2.10
[ 9065.911438] usb 3-1: New USB device strings: Mfr=1, Product=2, SerialNumber=3
[ 9065.911440] usb 3-1: Product: WCH-Link
[ 9065.911441] usb 3-1: Manufacturer: wch.cn
[ 9065.911442] usb 3-1: SerialNumber: 061D8F068AF0

```

-lsusb 
```
Bus 003 Device 003: ID 1a86:8012 QinHeng Electronics WCH-Link


```

- lsusb -v
```
Bus 003 Device 003: ID 1a86:8012 QinHeng Electronics WCH-Link
Device Descriptor:
  bLength                18
  bDescriptorType         1
  bcdUSB               2.00
  bDeviceClass          239 Miscellaneous Device
  bDeviceSubClass         2 [unknown]
  bDeviceProtocol         1 Interface Association
  bMaxPacketSize0        64
  idVendor           0x1a86 QinHeng Electronics
  idProduct          0x8012 WCH-Link
  bcdDevice            2.10
  iManufacturer           1 wch.cn
  iProduct                2 WCH-Link
  iSerial                 3 061D8F068AF0
  bNumConfigurations      1
  Configuration Descriptor:
    bLength                 9
    bDescriptorType         2
    wTotalLength       0x0062
    bNumInterfaces          3
    bConfigurationValue     1
    iConfiguration          0 
    bmAttributes         0x80
      (Bus Powered)
    MaxPower              500mA
    Interface Descriptor:
      bLength                 9
      bDescriptorType         4
      bInterfaceNumber        0
      bAlternateSetting       0
      bNumEndpoints           2
      bInterfaceClass       255 Vendor Specific Class
      bInterfaceSubClass      0 [unknown]
      bInterfaceProtocol      0 
      iInterface              4 WCH CMSIS-DAP
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x02  EP 2 OUT
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x83  EP 3 IN
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0
    Interface Association:
      bLength                 8
      bDescriptorType        11
      bFirstInterface         1
      bInterfaceCount         2
      bFunctionClass          2 Communications
      bFunctionSubClass       2 Abstract (modem)
      bFunctionProtocol       1 AT-commands (v.25ter)
      iFunction               4 WCH CMSIS-DAP
    Interface Descriptor:
      bLength                 9
      bDescriptorType         4
      bInterfaceNumber        1
      bAlternateSetting       0
      bNumEndpoints           2
      bInterfaceClass        10 CDC Data
      bInterfaceSubClass      0 [unknown]
      bInterfaceProtocol      0 
      iInterface              0 
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x03  EP 3 OUT
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x81  EP 1 IN
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0
    Interface Descriptor:
      bLength                 9
      bDescriptorType         4
      bInterfaceNumber        2
      bAlternateSetting       0
      bNumEndpoints           1
      bInterfaceClass         2 Communications
      bInterfaceSubClass      2 Abstract (modem)
      bInterfaceProtocol      1 AT-commands (v.25ter)
      iInterface              0 
      CDC Header:
        bcdCDC               1.10
      CDC Call Management:
        bmCapabilities       0x00
        bDataInterface          1
      CDC ACM:
        bmCapabilities       0x02
          line coding and serial state
      CDC Union:
        bMasterInterface        2
        bSlaveInterface         1 
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x84  EP 4 IN
        bmAttributes            3
          Transfer Type            Interrupt
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               1
Device Status:     0x0000
  (Bus Powered)


```
