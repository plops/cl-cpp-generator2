cmake_minimum_required(VERSION 3.10)  # Adjust if you need a different minimum version

project(risc_test CXX C ASM)  # Name your project

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-nostartfiles -static)
add_link_options(-nostartfiles -static)

# gcc --target-help
# shows possible options

# QingKe 32-bit RISC-V4C core: Supports RV32IMAC instruction set and
# self-extending instructions

add_compile_options( -march=rv32imac ) 

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


set(HAL_PATH  ${CMAKE_SOURCE_DIR}/../hal)

# Collect all C and assembly source files in the hal directory

#  I include all the files. They don't take long to compile and link
#  and I think the linker also removes unused code. Otherwise I would
#  have to manually add the files that are needed for whatever
#  functionality I want

file(GLOB_RECURSE HAL_SOURCES 
${HAL_PATH}/StdPeriphDriver/*.c
${HAL_PATH}/RVMSIS/*.c
${HAL_PATH}/Startup/*.S
${HAL_PATH}/board.c
)

# Source Files
add_executable(${PROJECT_NAME}
src/main.cpp 
${HAL_SOURCES}
)

# Include Directories
include_directories(
    ${HAL_PATH}/RVMSIS/
    ${HAL_PATH}/StdPeriphDriver/inc
    ${HAL_PATH}/User/
)

# Use the custom linker script to link the firmware
set(LINKER_SCRIPT ${HAL_PATH}/Ld/Link.ld)

target_link_options(${PROJECT_NAME} PRIVATE -T ${LINKER_SCRIPT})

# Libraries
target_link_libraries(${PROJECT_NAME} ${HAL_PATH}/StdPeriphDriver/libISP592.a)

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

