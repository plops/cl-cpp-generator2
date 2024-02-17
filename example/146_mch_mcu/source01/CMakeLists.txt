cmake_minimum_required(VERSION 3.10)  # Adjust if you need a different minimum version

project(risc_test)  # Name your project

# Toolchain and Compiler Settings
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)

set(SDK_PREFIX /home/martin/risc/)
set(RISCV_TOOLCHAIN_PATH ${SDK_PREFIX}/RISC-V_Embedded_GCC12/bin/)
set(OPENOCD_PATH ${SDK_PREFIX}/OpenOCD/bin/)

set(DEPENDENCY_PREFIX /home/martin/src/)
set(BROADCASTER_PATH  ${DEPENDENCY_PREFIX}/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/)

find_program(RISCV_GCC riscv-none-elf-g++ PATHS ${RISCV_TOOLCHAIN_PATH})

# Set compiler flags
set(CMAKE_C_FLAGS "-O0 -Wall") # Existing flags

# Source Files
add_executable(${PROJECT_NAME} src/main.cpp 
${BROADCASTER_PATH}/CH59x_pwr.c 
${BROADCASTER_PATH}/CH59x_gpio.c
${BROADCASTER_PATH}/CH59x_sys.c
${BROADCASTER_PATH}/CH59x_uart1.c)

# Include Directories
include_directories(
    ${BROADCASTER_PATH}/../RVMSIS/
    ${BROADCASTER_PATH}/../ble/LIB/
    ${BROADCASTER_PATH}/inc
)

# Libraries
target_link_libraries(${PROJECT_NAME}
${BROADCASTER_PATH}/../ble/LIB/LIBCH59xBLE.a
${BROADCASTER_PATH}/libISP592.a
)
