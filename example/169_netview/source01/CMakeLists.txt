cmake_minimum_required(VERSION 3.17)
project(cxxnet_client CXX)

option(NO_LIBS "Pull dependencies directly into project as source files" OFF)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)


if (NOT WIN32)
    add_compile_options(
            -Wall
            -Wextra
            -Wconversion
            -Wdouble-promotion
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-sign-conversion
            -Wvla
            -Wframe-larger-than=5000
            -Wstack-usage=10000
            #-Wshadow
            -Wformat=2
            -fno-common
            -fstack-usage
            -fvisibility=hidden
            -fno-strict-overflow
            -Wno-strict-overflow
    )

endif ()

set(SOURCE_FILES src/main.cpp
        src/VideoArchiveImpl.cpp
        src/VideoDecoder.cpp
        src/VideoDecoder.h
        src/Histogram.h
        src/DurationComputer.cpp
        src/DurationComputer.h
)

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
#    set(CMAKE_UNITY_BUILD true)
#    set(CMAKE_UNITY_BUILD_BATCH_SIZE 90)
    add_compile_options(
            -Ofast
            -ffunction-sections
            -fdata-sections
            # -Wpadded
            -march=znver3
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_link_options(
            #  -Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map
            -Wl,-gc-sections,--print-gc-sections
            #-Wl,-gc-sections
    )
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src)

set(capn /home/martin/src/capnproto/c++/src/capnp)
set(kj /home/martin/src/capnproto/c++/src/kj)
set(capnp_sources_lite
        ${capn}/c++.capnp.c++
        ${capn}/blob.c++
        ${capn}/arena.c++
        ${capn}/layout.c++
        ${capn}/list.c++
        ${capn}/any.c++
        ${capn}/message.c++
        ${capn}/schema.capnp.c++
        ${capn}/stream.capnp.c++
        ${capn}/serialize.c++
        ${capn}/serialize-packed.c++
)
set(capnp_sources_heavy
        ${capn}/schema.c++
        ${capn}/schema-loader.c++
        ${capn}/dynamic.c++
        ${capn}/stringify.c++
)
set(capnp-rpc_sources
        ${capn}/serialize-async.c++
        ${capn}/capability.c++
        ${capn}/membrane.c++
        ${capn}/dynamic-capability.c++
        ${capn}/rpc.c++
        ${capn}/rpc.capnp.c++
        ${capn}/rpc-twoparty.c++
        ${capn}/rpc-twoparty.capnp.c++
        ${capn}/persistent.capnp.c++
        ${capn}/ez-rpc.c++
)
set(kj_sources_lite
        ${kj}/array.c++
        ${kj}/cidr.c++
        ${kj}/list.c++
        ${kj}/common.c++
        ${kj}/debug.c++
        ${kj}/exception.c++
        ${kj}/io.c++
        ${kj}/memory.c++
        ${kj}/mutex.c++
        ${kj}/string.c++
        ${kj}/source-location.c++
        ${kj}/hash.c++
        ${kj}/table.c++
        ${kj}/thread.c++
        ${kj}/main.c++
        ${kj}/arena.c++
        ${kj}/test-helpers.c++
        ${kj}/units.c++
        ${kj}/encoding.c++
)
set(kj_sources_heavy
        ${kj}/refcount.c++
        ${kj}/string-tree.c++
        ${kj}/time.c++
        ${kj}/filesystem.c++
        ${kj}/filesystem-disk-unix.c++
        #${kj}/filesystem-disk-win32.c++
        ${kj}/parse/char.c++
)
set(kj-async_sources
        ${kj}/async.c++
        ${kj}/async-unix.c++
        #${kj}/async-win32.c++
        #${kj}/async-io-win32.c++
        ${kj}/async-io.c++
        ${kj}/async-io-unix.c++
        ${kj}/timer.c++
)

file(GLOB_RECURSE AV_SOURCES "/home/martin/src/avcpp/src/*.cpp")

if (NO_LIBS)
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
    find_package(FFmpeg
            COMPONENTS AVCODEC AVFORMAT AVUTIL AVDEVICE AVFILTER SWSCALE SWRESAMPLE REQUIRED)

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads)
    set(CMAKE_PREFIX_PATH /home/martin/vulkan)
    find_package(CapnProto REQUIRED)
    capnp_generate_cpp(videoSources videoHeaders proto/video.capnp)

    add_executable(
            ${PROJECT_NAME}
            ${SOURCE_FILES}
            ${videoSources}
            ${capnp_sources_lite}
            ${capnp_sources_heavy}
            ${capnp-rpc_sources}
            ${kj_sources_lite}
            ${kj_sources_heavy}
            ${kj-async_sources}
            ${AV_SOURCES}
    )
    target_compile_definitions(${PROJECT_NAME} PUBLIC __STDC_CONSTANT_MACROS)
    target_link_libraries(
            ${PROJECT_NAME}
            PRIVATE
            Threads::Threads
            FFmpeg::FFmpeg
    )
    target_include_directories(
            ${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR} # for autogenerated capnp header
            /home/martin/vulkan/include
            /home/martin/vulkan/include/avcpp/
    )
else ()
    set(CMAKE_PREFIX_PATH /home/martin/vulkan)
    find_package(CapnProto REQUIRED)
    capnp_generate_cpp(videoSources videoHeaders proto/video.capnp)
    # in ~/vulkan//lib64/cmake/avcpp/avcpp-config.cmake change this line
    # include("${CMAKE_CURRENT_LIST_DIR}/../../../lib/cmake/avcpp/avcpp-targets.cmake")
    find_package(avcpp REQUIRED)
    add_executable(
            ${PROJECT_NAME}
            ${SOURCE_FILES}
            ${videoSources}
    )
    target_link_libraries(
            ${PROJECT_NAME}
            PRIVATE
            avcpp::avcpp
            CapnProto::capnp-rpc
    )
    target_include_directories(
            ${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR} # for autogenerated capnp header
            /home/martin/vulkan/include
    )
endif ()






