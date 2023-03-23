cmake_minimum_required(VERSION 3.13)

# initialize the SDK based on PICO_SDK_PATH
# note: this must happen before project()
include(pico_sdk_import.cmake)

project(my_project)

set(CMAKE_CXX_STANDARD 17)

# initialize the Raspberry Pi Pico SDK
pico_sdk_init()


add_executable(hello_world
    main.cpp
)

# Add pico_stdlib library which aggregates commonly used features
target_link_libraries(hello_world pico_stdlib)

pico_enable_stdio_usb(hello_world 1)
pico_enable_stdio_uart(hello_world 0)
pico_generate_pio_header(hello_world ${CMAKE_CURRENT_LIST_DIR}/stepper.pio)
pico_generate_pio_header(hello_world ${CMAKE_CURRENT_LIST_DIR}/counter.pio)

# create map/bin/hex/uf2 file in addition to ELF.
pico_add_extra_outputs(hello_world)

