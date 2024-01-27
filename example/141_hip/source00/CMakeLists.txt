cmake_minimum_required(VERSION 3.28 FATAL_ERROR)  
project(vector_add_hip LANGUAGES CXX)  

# Set hipcc as the project's C++ compiler  
set(CMAKE_CXX_COMPILER hipcc)  


# Find HIP package  
find_package(HIP REQUIRED)  
  
# Set C++ standard  
set(CMAKE_CXX_STANDARD 17)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
  
# Include Google Test  
find_package(GTest REQUIRED)  
  
# Include directories for HIP and Google Test  
include_directories(${HIP_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
  
# Add HIP flags to CXX flags  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")  
  
# HIP source files  
set(HIP_SRC "src/vector_add.hip.cpp")  
  
# Convert HIP source files to C++ files  
add_library(vector_add_hip ${HIP_SRC})  
  
# Test executable  
add_executable(vector_add_test test/vector_add_test.cpp)  
target_link_libraries(vector_add_test GTest::GTest GTest::Main vector_add_hip)  
  
# Enable testing with CTest  
enable_testing()  
add_test(NAME VectorAddTest COMMAND vector_add_test)  


