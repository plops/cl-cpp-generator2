cmake_minimum_required(VERSION 3.12)
project(imgui_dsp)
set(CMAKE_CXX_STANDARD 20)

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
           # -fsanitize=undefined
	   -fsanitize=address
            #-fanalyzer
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
else ()
     set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS iostreams)
find_package(SoapySDR REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3_threads IMPORTED_TARGET)
find_package(OpenMP)


include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        src/GpsCACodeGenerator.cpp
        src/MemoryMappedComplexShortFile.cpp
        src/FFTWManager.cpp
        src/SdrManager.cpp
	src/ProcessMemoryInfo.cpp
        src/imgui.cpp
        src/imgui_draw.cpp
        src/imgui_impl_opengl3.cpp
        src/imgui_impl_glfw.cpp
        src/imgui_stdlib.cpp
        src/imgui_tables.cpp
        src/imgui_widgets.cpp
        src/implot.cpp
        src/implot_items.cpp
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        glfw
        OpenGL::GL
	Boost::iostreams
	PkgConfig::FFTW
	SoapySDR
	OpenMP::OpenMP_CXX
)

# Now you can build your app with
#     mkdir build && cd build && cmake .. && cmake --build .
