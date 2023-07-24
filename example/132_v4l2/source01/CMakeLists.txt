cmake_minimum_required(VERSION 3.12)
project(imgui_v4l2_project)
set(CMAKE_CXX_STANDARD 17)

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

##########################################################
# Prepare imgui_bundle during configure time
##########################################################
# Download imgui_bundle
#include(FetchContent)
#Set(FETCHCONTENT_QUIET FALSE)
#FetchContent_Declare(
#        imgui_bundle
#        GIT_REPOSITORY https://github.com/pthom/imgui_bundle.git
#        GIT_PROGRESS TRUE
#        # Enter the desired git tag below
#        GIT_TAG main
#)
#FetchContent_MakeAvailable(imgui_bundle)

# Make cmake function `imgui_bundle_add_app` available
#list(APPEND CMAKE_MODULE_PATH ${IMGUIBUNDLE_CMAKE_PATH})
#include(imgui_bundle_add_app)

# Uncomment the next line if you which to also automatically fetch and compile OpenCV for immvision support
set(IMMVISION_FETCH_OPENCV OFF)
set(IMGUI_BUNDLE_WITH_IMMVISION OFF)
set(IMGUI_OPENCV_CONAN OFF)

# find_package(SoapySDR REQUIRED)

##########################################################
# Build your app
##########################################################
# Call imgui_bundle_add_app
set(SOURCE_FILES
	src/main.cpp
	src/V4L2Capture.cpp
	)

include_directories(${CMAKE_SOURCE_DIR}/src)
#imgui_bundle_add_app(imgui_v4l2
#        src/main.cpp
#	src/V4L2Capture.cpp
#)

add_executable(imgui_v4l2 ${SOURCE_FILES})

# Now you can build your app with
#     mkdir build && cd build && cmake .. && cmake --build .
