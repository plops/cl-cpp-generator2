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



find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3_threads IMPORTED_TARGET)



add_executable(
        ${PROJECT_NAME}
	SearchFFT.cpp
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
	PkgConfig::FFTW
	
)
# Now you can build your app with
#     mkdir build && cd build && cmake .. && cmake --build .
