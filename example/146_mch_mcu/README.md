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
- even better might be the Examples/CH592/template, which seems to blink an LED 

```

- blue LED D2 indicates is connected to PA8 and 3v3. i think it will emit light when PA8 is gnd
- 32.768 kHz +/- 10ppm oscillator
- 32MHz +/- 10ppm oscillator
- key 1 on PB23 (rst)
- key 2 on PB22 (boot)


### Hardware Abstraction Layer (HAL) for CH592

- i think the best source for this code is the example that comes with MounRiverStudio:
MounRiver_Studio_Community_Linux_x64_V160/MRS_Community/template/wizard/WCH/RISC-V/CH59X/NoneOS/CH592F.zip (112kB)

```
cd ~/stage/cl-cpp-generator2/example/146_mch_mcu
mkdir hal
cd hal
unzip ../CH592F.zip
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

## Create hex file

- output of hex file generation visible here: https://youtu.be/L9Wrv7nW-S8?t=626

```

```

## Use OpenOCD to program the device

I am reading this article: https://robbert-groen.medium.com/getting-started-with-the-gd32vf103-risc-v-microcontroller-22cb34718b0d


### The programmer
- user manual: https://www.wch-ic.com/downloads/WCH-LinkUserManual_PDF.html
- how to connect to CH592
  - swdio <-> PB14
  - swclk <-> PB15
  - also 3v3 and gnds
  
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


- install udev stuff

```
cd /home/martin/risc/beforeinstall
cp 50-wch.rules /etc/udev/rules.d/
cp 60-openocd.rules /etc/udev/rules.d/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/martin/risc/beforeinstall/
# /home/martin/risc/OpenOCD/bin/openocd -h
Open On-Chip Debugger 0.11.0+dev-02415-gfad123a16-dirty (2023-10-11-14:01)
Licensed under GNU GPL v2
For bug reports, read
        http://openocd.org/doc/doxygen/bugs.html
Open On-Chip Debugger
Licensed under GNU GPL v2
--help       | -h       display this help
--version    | -v       display OpenOCD version
--file       | -f       use configuration file <name>
--search     | -s       dir to search for config files and scripts
--debug      | -d       set debug level to 3
             | -d<n>    set debug level to <level>
--log_output | -l       redirect log output to file <name>
--command    | -c       run <command>


```
- i watch this video: https://www.youtube.com/watch?v=L9Wrv7nW-S8 EEVblog 1524 - The 10 CENT RISC V Processor! CH32V003

- i found chinese documentation of openocd usage in MounRiverStudio Community

- i translated it with gemini and placed it in doc/README_openocd.md

```
sudo su
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/martin/risc/beforeinstall
export PATH=$PATH:/home/martin/risc/OpenOCD/bin/:/home/martin/risc/RISC-V_Embedded_GCC12/bin

openocd -f /home/martin/risc/OpenOCD/bin/wch-riscv.cfg -c init -c halt -c "flash erase_sector wch_riscv 0 last " -c exit



cd /home/martin/stage/cl-cpp-generator2/example/146_mch_mcu/source01
openocd -f wch-riscv.cfg -c init -c halt -c "program b/risc_test " -c exit
```

- not working

- i tried some sample projects in MounRiverStudio. i think i managed
  to compile the adc example but downloading wanted to update the
  programmer. which may not have worked:

```
20:16:02:245 >> Attempt to open link device and upgrade firmware if necessary...
20:16:07:153 >> WCH-Link not found.
```

- now lsusb shows

```
Bus 003 Device 009: ID 4348:55e0 WinChipHead 

```

- and lsusb -v:

```
Bus 003 Device 018: ID 4348:55e0 WinChipHead 
Couldn't open device, some information will be missing
Device Descriptor:
  bLength                18
  bDescriptorType         1
  bcdUSB               1.10
  bDeviceClass          255 Vendor Specific Class
  bDeviceSubClass       128 [unknown]
  bDeviceProtocol        85 
  bMaxPacketSize0         8
  idVendor           0x4348 WinChipHead
  idProduct          0x55e0 
  bcdDevice            1.00
  iManufacturer           0 
  iProduct                0 
  iSerial                 0 
  bNumConfigurations      1
  Configuration Descriptor:
    bLength                 9
    bDescriptorType         2
    wTotalLength       0x002e
    bNumInterfaces          1
    bConfigurationValue     1
    iConfiguration          0 
    bmAttributes         0x80
      (Bus Powered)
    MaxPower              100mA
    Interface Descriptor:
      bLength                 9
      bDescriptorType         4
      bInterfaceNumber        0
      bAlternateSetting       0
      bNumEndpoints           4
      bInterfaceClass       255 Vendor Specific Class
      bInterfaceSubClass    128 [unknown]
      bInterfaceProtocol     85 
      iInterface              0 
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x82  EP 2 IN
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0
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
        bEndpointAddress     0x81  EP 1 IN
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0
      Endpoint Descriptor:
        bLength                 7
        bDescriptorType         5
        bEndpointAddress     0x01  EP 1 OUT
        bmAttributes            2
          Transfer Type            Bulk
          Synch Type               None
          Usage Type               Data
        wMaxPacketSize     0x0040  1x 64 bytes
        bInterval               0

```
### Risc-V Processor Manual

https://www.wch-ic.com/downloads/QingKeV4_Processor_Manual_PDF.html

## A Rust Tool to program using WCH Link

https://github.com/ch32-rs/wlink

## Programming via USB ISP

- https://github.com/jmaselbas/wch-isp (C)
- https://github.com/ch32-rs/wchisp/releases/tag/nightly (Rust)


- press the boot button on the CH592 while plugging in the
  connector. for a while (60sec) it seems to be registered as a usb device:
  
- this is the kmsg output
```
c<6>[26405.147714] usb 3-1: new full-speed USB device number 27 using xhci_hcd
<6>[26405.295775] usb 3-1: New USB device found, idVendor=4348, idProduct=55e0, bcdDevice=23.00
<6>[26405.295780] usb 3-1: New USB device strings: Mfr=0, Product=0, SerialNumber=0
<6>[26467.621093] usb 3-1: USB disconnect, device number 27

```

- this is the output of the rust wchisp tool
```
archlinux /dev/shm # ./wchisp info
23:54:34 [INFO] Chip: CH592[0x9222] (Code Flash: 448KiB, Data EEPROM: 32KiB)
23:54:34 [INFO] Chip UID: DA-36-4F-10-53-5C-7C-A3
23:54:34 [INFO] BTVER(bootloader ver): 02.30
23:54:34 [INFO] Current config registers: ffffffffffffffff4d0fff4f00020300da364f10535c7ca3
RESERVED: 0xFFFFFFFF
WPROTECT: 0xFFFFFFFF
  [0:0]   NO_KEY_SERIAL_DOWNLOAD 0x1 (0b1)
    `- Enable
  [1:1]   DOWNLOAD_CFG 0x1 (0b1)
    `- PB22(Default set)
USER_CFG: 0x4FFF0F4D
  [2:0]   RESERVED 0x5 (0b101)
    `- Default
  [3:3]   CFG_RESET_EN 0x1 (0b1)
    `- Enable
  [4:4]   CFG_DEBUG_EN 0x0 (0b0)
    `- Disable
  [5:5]   RESERVED 0x0 (0b0)
    `- Default
  [6:6]   CFG_BOOT_EN 0x1 (0b1)
    `- Enable
  [7:7]   CFG_ROM_READ 0x0 (0b0)
    `- Disable the programmer to read out, and keep the program secret
  [27:8]  RESERVED 0xFFF0F (0b11111111111100001111)
    `- Error
  [31:28] VALID_SIG 0x4 (0b100)
    `- Valid
```


- erase the flash:

```

146_mch_mcu/source01 # ./wchisp -v erase
00:00:50 [DEBUG] (1) wchisp::transport::usb: Found USB Device Bus 003 Device 032: ID 4348:55e0
00:00:50 [DEBUG] (1) wchisp::transport: => a11200   00004d4355204953502026205743482e434e
00:00:50 [DEBUG] (1) wchisp::transport: <= a1000200 9222
00:00:50 [DEBUG] (1) wchisp::transport: => a11200   00004d4355204953502026205743482e434e
00:00:50 [DEBUG] (1) wchisp::transport: <= a1000200 9222
00:00:50 [DEBUG] (1) wchisp::flashing: found chip: CH592[0x9222]
00:00:50 [DEBUG] (1) wchisp::transport: => a70200   1f00
00:00:50 [DEBUG] (1) wchisp::transport: <= a7001a00 1f00ffffffffffffffff4d0fff4f00020300da364f10535c7ca3
00:00:50 [DEBUG] (1) wchisp::flashing: read_config: ffffffffffffffff4d0fff4f00020300da364f10535c7ca3
00:00:50 [DEBUG] (1) wchisp::transport: => a40400   c0010000
00:00:51 [DEBUG] (1) wchisp::transport: <= a4000200 0000
00:00:51 [INFO] Erased 448 code flash sectors

```

- write program
```
146_mch_mcu/source01 # ./wchisp  flash b/risc_test
00:02:49 [INFO] Chip: CH592[0x9222] (Code Flash: 448KiB, Data EEPROM: 32KiB)
00:02:49 [INFO] Chip UID: DA-36-4F-10-53-5C-7C-A3
00:02:49 [INFO] BTVER(bootloader ver): 02.30
00:02:49 [INFO] Current config registers: ffffffffffffffff4d0fff4f00020300da364f10535c7ca3
RESERVED: 0xFFFFFFFF
WPROTECT: 0xFFFFFFFF
...
  [31:28] VALID_SIG 0x4 (0b100)
    `- Valid
00:02:49 [INFO] Read b/risc_test as ELF format
00:02:49 [INFO] Found loadable segment, physical address: 0x00010000, virtual address: 0x00010000, flags: 0x5
00:02:49 [INFO] Section names: [".text", ".highcode", ".rodata"]
00:02:49 [INFO] Found loadable segment, physical address: 0x0001202c, virtual address: 0x0001202c, flags: 0x6
00:02:49 [INFO] Section names: [".eh_frame", ".init_array", ".fini_array", ".data", ".sdata"]
00:02:49 [INFO] Firmware size: 10240
00:02:49 [INFO] Erasing...
00:02:49 [INFO] Erased 11 code flash sectors
00:02:50 [INFO] Erase done
00:02:50 [INFO] Writing to code flash...
██████████████████████████████████████████████████████████████████████████████████████████████████████ 10240/1024000:02:50 [INFO] Code flash 10240 bytes written
00:02:51 [INFO] Verifying...
██████████████████████████████████████████████████████████████████████████████████████████████████████ 10240/1024000:02:51 [INFO] Verify OK
00:02:51 [INFO] Now reset device and skip any communication errors
00:02:51 [INFO] Device reset

```


## BLE Library

The library for bluetooth is not contained in MounRiverStudio or the SDK.
Only the PCB supplier has this library in the example programs:

```
martin@archlinux ~/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble $ find . -type f
./LIB/CH59xBLE_ROM_PERI.hex
./LIB/LIBCH59xBLE.a
./LIB/ble_task_scheduler.S
./LIB/CH59xBLE_ROM.h
./LIB/CH59xBLE_ROM.hex
./LIB/CH59xBLE_LIB.h
./LIB/CH59xBLE_ROMx.hex
./HAL/KEY.c
./HAL/RTC.c
./HAL/MCU.c
./HAL/LED.c
./HAL/SLEEP.c
./HAL/include/SLEEP.h
./HAL/include/KEY.h
./HAL/include/LED.h
./HAL/include/HAL.h
./HAL/include/RTC.h
./HAL/include/CONFIG.h
./Profile/devinfoservice.c
./Profile/include/devinfoservice.h
./APP/broadcaster.c
./APP/include/broadcaster.h
martin@archlinux ~/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble $ du -sh
3.1M    .

```

I'm reluctant to add 3MB of binary files to my repository.

A CONFIG.h file seems to contain a lot of bluetooth related configurations:
```
#ifdef CH59xBLE_ROM
#include "CH59xBLE_ROM.h"
#else
#include "CH59xBLE_LIB.h"
#endif

```

The file has some comments in chinese. Here is Gemini Advanced 1.0 translation:

**MAC**
*BLE_MAC* - Whether to customize the Bluetooth MAC address (Default: FALSE - Uses chip's built-in MAC address). Requires modifying MAC address definition in main.c

**DCDC**
*DCDC_ENABLE* - Whether to enable DCDC (Default: FALSE) 

**SLEEP**
*HAL_SLEEP* - Whether to enable sleep functionality (Default: FALSE)
*SLEEP_RTC_MIN_TIME* - Minimum time for sleep mode (in RTC clock cycles)
*SLEEP_RTC_MAX_TIME* - Maximum time for sleep mode (in RTC clock cycles)
*WAKE_UP_RTC_MAX_TIME* - Wait time for 32MHz crystal oscillator stabilization (in RTC clock cycles). Different values used for different sleep modes:
   * Sleep Mode/Power Down Mode - 45 (Default)
   * Pause Mode - 45
   * Idle Mode - 5

**TEMPERATION**
*TEM_SAMPLE* -  Whether to enable temperature-based calibration function. Single calibration takes less than 10ms (Default: TRUE)

**CALIBRATION**
*BLE_CALIBRATION_ENABLE* -  Whether to enable the periodic calibration function. Single calibration takes less than 10ms (Default: TRUE)
*BLE_CALIBRATION_PERIOD* - Calibration period in milliseconds (Default: 120000)

**SNV**
*BLE_SNV* - Whether to enable SNV functionality for storing bonding information (Default: TRUE)
*BLE_SNV_ADDR* - SNV information storage address, using the last block of data flash (Default: 0x77E00)
*BLE_SNV_BLOCK*  - Size of SNV storage block (Default: 256)
*BLE_SNV_NUM* - Number of SNV storage blocks (Default: 1)

**RTC**
*CLK_OSC32K* - RTC clock selection. Must use an external 32KHz crystal oscillator for the host role.  (0: External (32768Hz), Default: 1: Internal (32000Hz), 2: Internal (32768Hz))

**MEMORY**
*BLE_MEMHEAP_SIZE* -  RAM size used by the Bluetooth protocol stack (must be at least 6K (Default: (1024*6))

**DATA**
*BLE_BUFF_MAX_LEN* - Maximum packet length for single connection (Default: 27 (ATT_MTU=23), Range: [27~516])
*BLE_BUFF_NUM* - Number of packets the controller buffers (Default: 5)
*BLE_TX_NUM_EVENT* - Maximum number of data packets that can be sent for a single connection event (Default: 1)
*BLE_TX_POWER* - Transmit power (Default: LL_TX_POWEER_0_DBM (0dBm))

**MULTICONN**
*PERIPHERAL_MAX_CONNECTION* - Maximum number of peripheral connections (Default: 1)
*CENTRAL_MAX_CONNECTION* - Maximum number of central connections (Default: 3)

#define CLK_OSC32K     1  // Comment stating implications of changing this definition  

**Explanation**

These settings configure a Bluetooth Low Energy (BLE) chip. Here's a breakdown of the concepts:

* **MAC Address:** Unique identifier for the BLE device.
* **DCDC:** A voltage regulator for improved power efficiency.
* **Sleep:**  Low-power modes for energy conservation.
* **Calibration:** Ensures timing accuracy, especially across temperature changes.
* **SNV:** Non-volatile storage for pairing/bonding information.
* **RTC:** Real-time clock for timekeeping (32KHz clock source selection is important).
* **Memory:** RAM allocation for the BLE stack.
* **Data:**  Packet sizes and buffering settings.
* **Multi-connection:** How many simultaneous BLE connections are supported.

