cmake_minimum_required(VERSION 3.10)

project(my_project)

# If you want to use C++17 replace 14 with 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
# set your source files
set(SOURCE_FILES main.cpp ArgException.cpp)

# set compiler flags

add_compile_options(
    -g -g3 -ggdb -gdwarf-4
    -Wall -Wextra -Wconversion -Wdouble-promotion
    -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
    -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
    -Wshadow
    -fvisibility=hidden
    -fno-strict-overflow -Wno-strict-overflow
    -fsanitize=undefined  -fanalyzer
)

find_package(SoapySDR REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})


target_link_libraries(${PROJECT_NAME} SoapySDR)
