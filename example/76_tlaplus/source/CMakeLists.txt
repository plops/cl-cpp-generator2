# sudo dnf install fmt-devel
cmake_minimum_required(VERSION 3.4)
project(consumer_producer LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED yes)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# UndefinedBehaviorSanitizer
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)
if (ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif ()

# ThreadSanitizer
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif ()

# LeakSanitizer
option(ENABLE_LSAN "Enable LeakSanitizer" ON)
if (ENABLE_LSAN)
    add_compile_options(-fsanitize=leak)
    add_link_options(-fsanitize=leak)
endif ()

option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
if (ENABLE_MSAN)
    add_compile_options(-fsanitize=memory)
    add_link_options(-fsanitize=memory)
endif ()


#set( CMAKE_CXX_COMPILER clang++ )
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0  -fno-omit-frame-pointer ")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0   ")
set(SRCS /home/martin/stage/cl-cpp-generator2/example/76_tlaplus/source/main.cpp)
add_executable(consumer_producer ${SRCS})
target_compile_features(consumer_producer PUBLIC cxx_std_20)
find_package(fmt CONFIG REQUIRED)
target_link_libraries(consumer_producer PRIVATE fmt)
