cmake_minimum_required(VERSION 3.12)
project(imgui_dsp)
set(CMAKE_CXX_STANDARD 23)

# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
            -Wall -Wextra -Wconversion -Wdouble-promotion
            -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
            -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
            -Wshadow
            -fvisibility=hidden
            -fno-strict-overflow -Wno-strict-overflow
            -fsanitize=undefined
            #-fanalyzer
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
endif ()




include_directories(${CMAKE_SOURCE_DIR}/src)
add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        src/GpsCACodeGenerator.cpp
        src/imgui.cpp
        src/imgui_draw.cpp
        src/imgui_impl_opengl3.cpp
        src/imgui_stdlib.cpp
        src/imgui_tables.cpp
        src/imgui_widgets.cpp
        src/implot.cpp
        src/implot_items.cpp
)

# target_link_libraries(imgui_soapysdr PRIVATE SoapySDR)

# Now you can build your app with
#     mkdir build && cd build && cmake .. && cmake --build .
