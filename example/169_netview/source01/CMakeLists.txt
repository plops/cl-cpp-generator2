cmake_minimum_required(VERSION 3.17)
project(cxxnet CXX)

#set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#find_package(FFmpeg)

set(CMAKE_PREFIX_PATH /home/martin/vulkan)
find_package(CapnProto CONFIG REQUIRED)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)


if (NOT WIN32)
    add_compile_options(
            -Wall -Wextra -Wconversion -Wdouble-promotion
            -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
            -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
            -Wshadow -Wformat=2 -fno-common -fstack-usage
            -fvisibility=hidden
            -fno-strict-overflow -Wno-strict-overflow
    )

endif ()
set(SOURCE_FILES src/main.cpp)

# AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# UndefinedBehaviorSanitizer
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
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
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
if (ENABLE_LSAN)
    add_compile_options(-fsanitize=leak)
    add_link_options(-fsanitize=leak)
endif ()


# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -Og -g -g3 -ggdb -gdwarf-4
    )
else ()
    add_compile_options(
            -Ofast
            -ffunction-sections -fdata-sections
            -Wpadded
            -march=znver3
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_link_options(-Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src)

capnp_generate_cpp(videoSources videoHeaders proto/video.capnp)

add_executable(
        ${PROJECT_NAME}
        ${SOURCE_FILES}
        ${videoSources}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        CapnProto::capnp
        avformat
)