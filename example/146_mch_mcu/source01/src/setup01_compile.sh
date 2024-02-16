#!/usr/bin/bash

export PATH=/home/martin/risc/RISC-V_Embedded_GCC12/bin/:/home/martin/risc/OpenOCD/bin/:$PATH

riscv-none-elf-g++ \
    -O0 -Wall \
    -o main \
    -I/home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/RVMSIS/ \
    -I/home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble/LIB/ \
    -I/home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/inc \
    main.cpp \
    /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/CH59x_pwr.c \
    /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/CH59x_gpio.c \
    /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/CH59x_sys.c \
    /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/CH59x_uart1.c \
    /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble/LIB/LIBCH59xBLE.a \
    /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/libISP592.a 
