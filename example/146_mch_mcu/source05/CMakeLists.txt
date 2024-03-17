cmake_minimum_required(VERSION 3.10)  # Adjust if you need a different minimum version

project(risc_usb_test CXX C ASM)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# create a cmake option to indicate the build is for the target
# this is enabled when building for CH592 embedded system
# without this option we can build code for amd64 (just for testing)
option(BUILD_FOR_TARGET "Build for target" OFF)

# if the build is for the target, we need these flags
if (BUILD_FOR_TARGET)

  # Compilation/Linking Flags

  # -static: Create a statically linked binary (all dependencies included).
  # -nostartfiles: Skip default system startup files for full control.
  # -fno-exceptions: Disable C++ exception handling (reduces size).
  # -fno-rtti: Disable C++ Run-Time Type Information (reduces size).

  # fmt Library Configuration https://github.com/fmtlib/fmt/issues/1617

  # -DFMT_USE_FLOAT=0: Disable float formatting support.
  # -DFMT_USE_DOUBLE=0: Disable double formatting support.
  # -DFMT_USE_LONG_DOUBLE=0: Disable long double formatting support.
  # -DFMT_STATIC_THOUSANDS_SEPARATOR: Enforce a fixed thousands separator character.

  set(EMBED_OPTIONS
    -static 
    -nostartfiles
    -fno-exceptions
    -fno-rtti
    -DFMT_USE_FLOAT=0
    -DFMT_USE_DOUBLE=0
    -DFMT_USE_LONG_DOUBLE=0
    -DFMT_STATIC_THOUSANDS_SEPARATOR
    -DFMT_HEADER_ONLY
  )
  add_compile_options(
    ${EMBED_OPTIONS}
  )
  add_link_options(
    ${EMBED_OPTIONS}
  )

  # gcc --target-help
  # shows possible options

  # QingKe 32-bit RISC-V4C core: Supports RV32IMAC instruction set and
  # self-extending instructions

  #add_compile_options( -march=rv32imac -mabi=ilp32 -msmall-data-limit=8 -mno-save-restore ) 
  #add_link_options(-march=rv32imac -mabi=ilp32)


  #   -march=rv32imac_zicsr_zifencei: Targets the RISC-V 32-bit architecture with the following extensions:
  #   IMAC: Integer multiply-accumulate, atomic, and compressed instructions
  #   Zicsr: Control and Status Register (CSR) instructions
  #   Zifencei: Fence instructions for memory ordering
  #   -mabi=ilp32: Specifies the 32-bit Integer, Long, and Pointer Application Binary Interface (ABI). This determines how data is represented and how functions interact in memory.
  #   -msmall-data-limit=8: Sets a limit of 8 bytes for the small data area in the program. This can optimize memory access for frequently used data.
  #   -mno-save-restore: Instructs the compiler to avoid automatically saving and restoring registers when calling functions. This can provide a performance increase, but requires careful manual register management if needed.
  
  
  
  add_compile_options( -march=rv32imac_zicsr_zifencei -mabi=ilp32 -msmall-data-limit=8 -mno-save-restore ) 


  #   -march=rv32imac_zicsr_zifencei -mabi=ilp32: Enforces consistency with the compilation flags, ensuring the linker generates code for the RISC-V target and ABI.

  add_link_options(-march=rv32imac_zicsr_zifencei -mabi=ilp32)

  #   -Xlinker --gc-sections: Enables "garbage collection" at the linker level. This removes any unused code sections from the final executable, reducing size.

  add_link_options(-Xlinker --gc-sections)
  

  #   --specs=nano.specs --specs=nosys.specs: Uses custom linker scripts.

  #   nano.specs: configuration for extremely small executables, potentially tailored to the Newlib C library.
  #   nosys.specs: prevents linking against standard system libraries, implying a bare-metal environment.

  add_link_options(
    --specs=nano.specs 
    --specs=nosys.specs
  )

endif()

# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(
    -g -g3 -ggdb -gdwarf-4
  )


  add_link_options(-Wl,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)

else ()
  add_compile_options(
    #-O0
    -Os #-ffunction-sections -fdata-sections # don't use together with flto
  )
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # flto


  #   -gc-sections: Enables "garbage collection" at the linker stage. The linker will analyze your code and discard any unused functions or data sections to reduce the final executable size.
  #   --print-gc-sections: Instructs the linker to generate a report detailing which sections were removed during the garbage collection process. This can help you pinpoint potential areas for further size optimization. 
  #   --print-memory-usage: Tells the linker to provide a summary of the memory usage of your executable, broken down into categories like code, data, etc. This helps analyze memory consumption.
  #   -Map=...: Generates a map file that details the memory layout of your linked executable.
  #   --cref: Enables the generation of a cross-reference table. This table provides information about how symbols (functions, variables) are defined and used throughout your codebase.

  add_link_options(-Wl,-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map,--cref)
endif ()


set(HAL_PATH  ${CMAKE_SOURCE_DIR}/../hal)

# Collect all C and assembly source files in the hal directory

#  I include all the files. They don't take long to compile and link
#  and I think the linker also removes unused code. Otherwise I would
#  have to manually add the files that are needed for whatever
#  functionality I want

file(GLOB_RECURSE HAL_SOURCES
  ${HAL_PATH}/StdPeriphDriver/*_sys.c
  ${HAL_PATH}/StdPeriphDriver/*_uart*.c
  #${HAL_PATH}/StdPeriphDriver/*_clk.c
  #${HAL_PATH}/StdPeriphDriver/*_adc.c
  #${HAL_PATH}/StdPeriphDriver/*_pwr.c
  ${HAL_PATH}/StdPeriphDriver/*_gpio.c
  #${HAL_PATH}/StdPeriphDriver/*_usbdev.c
  #${HAL_PATH}/RVMSIS/*.c
  ${HAL_PATH}/Startup/*.S
  #${HAL_PATH}/board.c
  #${HAL_PATH}/../BLE/HAL/*.c
  #${HAL_PATH}/../BLE/LIB/*.S
)

# Separately define my sourcefiles and those of the HAL. I want to
# emit rigorous warnings for my code.

# Source Files
add_library(mine 
  src/Ch592UsbRegisters.cpp   
  src/Ch592UsbRegisters.h
  src/UsbDeviceDescriptor.cpp
  src/UsbDeviceDescriptor.h
  src/UsbConfigurationDescriptor.cpp
  src/UsbConfigurationDescriptor.h
  src/Uart.cpp
  src/Uart.h
)

# add_library(fmt
#   /home/martin/src/fmt/src/format.cc
#   /home/martin/src/fmt/src/os.cc
#   /home/martin/src/fmt/src/fmt.cc
#   )


# -Wall: Enables a set of warning messages for potential issues in the code.
# -Wextra: Enables additional warning messages beyond what -Wall provides.
# -Wconversion: Warns about implicit type conversions that may result in data loss.
# -Wdouble-promotion: Warns about implicit promotion of float values to double.
# -Wno-unused-parameter: Suppresses warnings for unused function parameters.
# -Wno-unused-function: Suppresses warnings for unused functions.
# -Wno-sign-conversion: Suppresses warnings for implicit sign conversions.
# -Wvla: Warns about variable-length arrays, which may have limitations.
# -Wframe-larger-than=5000: Warns if a function's stack frame size exceeds 5000 bytes.
# -Wstack-usage=10000: Warns if the stack usage of a function exceeds 10000 bytes.
# -Wshadow: Warns about variable shadowing, where a local variable hides another variable with the same name.
# -Wformat=2: Enables additional format string checking for printf-style functions.
# -fno-common: Disallows the use of tentative definitions and forces each global variable to have a unique definition.
# -fstack-usage: Generates a report of stack usage for each function.

target_compile_options(
  mine PRIVATE
  #-Werror
  -Wall -Wextra -Wconversion -Wdouble-promotion
  -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
  -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
  -Wshadow -Wformat=2 -fno-common -fstack-usage
  -fvisibility=hidden
  -fno-strict-overflow -Wno-strict-overflow
  #-Wpadded
  -fdiagnostics-color=always
)

# if we build for target set make sure the pre-processor
# define BUILD_FOR_TARGET is specified
if (BUILD_FOR_TARGET)
  target_compile_definitions(mine PRIVATE BUILD_FOR_TARGET)

  # Include Directories
  include_directories(
    ${HAL_PATH}/RVMSIS/
    ${HAL_PATH}/StdPeriphDriver/inc
    # git clone https://github.com/fmtlib/fmt
    /home/martin/src/fmt/include/
    #   ${HAL_PATH}/User/
    #   ${HAL_PATH}/../BLE/HAL/include
    #   ${HAL_PATH}/../BLE/LIB
  )

  # Use the custom linker script to link the firmware
  set(LINKER_SCRIPT ${HAL_PATH}/Ld/Link.ld)

  add_executable(${PROJECT_NAME}
    src/main.cpp
    ${HAL_SOURCES}
  )

  target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_FOR_TARGET)
  target_link_options(${PROJECT_NAME} PRIVATE -T ${LINKER_SCRIPT})



else()

  add_executable(${PROJECT_NAME}
    src/main.cpp
  )
endif()

target_link_libraries(${PROJECT_NAME}
  mine 
)

if (BUILD_FOR_TARGET)

  # Libraries
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(${PROJECT_NAME}
      ${HAL_PATH}/StdPeriphDriver/libISP592.a
      mine
    )
  else()
  # when the ${PROJECT_NAME} has been build, strip the binary:
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND /home/martin/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-nm -S --size-sort -C -l ${PROJECT_NAME} > ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.nm
    COMMAND  /home/martin/risc/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-strip ${PROJECT_NAME}
  )
  endif ()
endif()
