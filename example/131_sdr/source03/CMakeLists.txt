cmake_minimum_required(VERSION 3.12)
project(imgui_dsp)

set(CMAKE_CXX_STANDARD 20)

option(USE_ADDRESS_SANITIZER "Use Address Sanitizer" OFF)

# https://devtalk.blender.org/t/speed-up-c-compilation/30508
# set the mold linker
set(CMAKE_LINKER mold)

# set ccache for C and C++ compilers
#set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(CXX clang++)

# create compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

if(USE_ADDRESS_SANITIZER)
    add_compile_options(-fsanitize=address)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

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
    )
else ()
     set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif ()



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
        src/imgui.cpp
        src/imgui_draw.cpp
        src/imgui_impl_opengl3.cpp
        src/imgui_impl_glfw.cpp
        src/imgui_stdlib.cpp
        src/imgui_tables.cpp
        src/imgui_widgets.cpp
        src/implot.cpp
        src/implot_items.cpp
        src/main.cpp
        src/GpsCACodeGenerator.cpp
        src/MemoryMappedComplexShortFile.cpp
        src/FFTWManager.cpp
        src/SdrManager.cpp
        src/GpsTracker.cpp
	src/ProcessMemoryInfo.cpp
)

#set(CMAKE_UNITY_BUILD_BATCH_SIZE 10)

# enable Unity build
#set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON)


# exclude main.cpp from Unity build (this is useful if main.cpp is the only file that I edit right now)
#set_source_files_properties(src/main.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

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
