cmake_minimum_required(VERSION 3.10)  # Adjust if you need a different minimum version

project(risc_test CXX C)  # Name your project

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create a CMake Option to enable adress sanitizer (not supported by sdk)
#option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
#if(ENABLE_ASAN)
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
#        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
#endif()



# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
    )
else ()
    add_compile_options(
            #-O0
            -O3 -ffunction-sections -fdata-sections # don't use together with flto
    )
    #set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # flto
    add_link_options(-Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)
endif ()




set(SDK_PREFIX /home/martin/risc/)
set(RISCV_TOOLCHAIN_PATH ${SDK_PREFIX}/RISC-V_Embedded_GCC12/bin/)
set(OPENOCD_PATH ${SDK_PREFIX}/OpenOCD/bin/)

set(DEPENDENCY_PREFIX /home/martin/src/)
set(BROADCASTER_PATH  ${DEPENDENCY_PREFIX}/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/)

# Source Files
add_executable(${PROJECT_NAME}
src/main.cpp 
${BROADCASTER_PATH}/CH59x_pwr.c 
${BROADCASTER_PATH}/CH59x_gpio.c
${BROADCASTER_PATH}/CH59x_sys.c
${BROADCASTER_PATH}/CH59x_uart1.c
${BROADCASTER_PATH}/CH59x_adc.c
${BROADCASTER_PATH}/../ble/HAL/LED.c
${BROADCASTER_PATH}/../ble/HAL/MCU.c
${BROADCASTER_PATH}/../ble/HAL/KEY.c
${BROADCASTER_PATH}/../ble/APP/broadcaster.c
)

# Include Directories
include_directories(
    ${BROADCASTER_PATH}/../RVMSIS/
    ${BROADCASTER_PATH}/../ble/LIB/
    ${BROADCASTER_PATH}/inc
    ${BROADCASTER_PATH}/../ble/APP/include/
    ${BROADCASTER_PATH}/../ble/HAL/include/
    ${BROADCASTER_PATH}/../ble/Profile/include/
)

# Libraries
target_link_libraries(${PROJECT_NAME}
${BROADCASTER_PATH}/../ble/LIB/LIBCH59xBLE.a
${BROADCASTER_PATH}/libISP592.a
)

target_compile_options(
        ${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wconversion -Wdouble-promotion
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
        -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
        -Wshadow -Wformat=2 -fno-common -fstack-usage
        -fvisibility=hidden
        -fno-strict-overflow -Wno-strict-overflow
        #-Wpadded
)

