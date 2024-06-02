# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(packet_listener4)

# Specify C++17 standard
set(CMAKE_CXX_STANDARD 23)

# Find necessary libraries (this example is minimal, you might need more)
# You can use find_package for more complex dependencies.
#find_package(Threads REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Link libraries
#target_link_libraries(packet_listener Threads::Threads) 

# Add compiler options (optional)
# For example, to enable warnings:
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)

# Define the test executable
option(ENABLE_CAPTURE_TESTS "Enable unit tests for the network capturing." OFF)

if(ENABLE_CAPTURE_TESTS)
        enable_testing()

        set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")
        #include(coverage)
        #add_coverage_target("*/tests/*")

        # Fetch Google Test
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0  # specify the commit hash or tag here
        )
        FetchContent_MakeAvailable(googletest)

        set(test_sources
	        tests/test_recv.cpp
                tests/test_rx_packet_mmap.cpp
                src/main.cpp
        )
        set_source_files_properties(${test_sources} PROPERTIES UNITY_GROUP "Tests")

        add_executable(unit_tests
                ${test_sources}
        )    
        
        # Set specific compiler options for the unit_tests target (turn of optimizations for coverage)
	#target_compile_options(unit_tests PRIVATE
        #        -O0 -fprofile-arcs -ftest-coverage --coverage)  
        #target_link_options(unit_tests PRIVATE --coverage)  # Link with the coverage library

        # Link the test executable with Google Test and the necessary libraries
        target_link_libraries(unit_tests gtest_main)
        target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

        # Register the test with CTest  
        add_test(NAME unit_tests COMMAND unit_tests)  
endif()
