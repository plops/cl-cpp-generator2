cmake_minimum_required(VERSION 3.16)
project(ModernCppPools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Google Test ---
#include(FetchContent)
#FetchContent_Declare(
#  googletest
#  GIT_REPOSITORY https://github.com/google/googletest.git
#  GIT_TAG       release-1.14.0 # Or a specific commit
#)
#FetchContent_MakeAvailable(googletest)

# --- Application ---
add_executable(pool_app
    src/app/main.cpp
    # Add other .cpp files if they exist (most are header-only in this example)
)
target_compile_features(pool_app PRIVATE cxx_std_20)
if(WIN32)
    # Link against Ws2_32.lib for htons/ntohs on Windows if not using a cross-platform lib
    # For MinGW, it might be linked automatically or need -lws2_32
    # For MSVC, this is typically needed if <winsock2.h> is used directly.
    # However, arpa/inet.h is POSIX. On Windows, you might need a different approach or a library.
    # For simplicity, assuming POSIX or MinGW where arpa/inet.h might pull ws2_32.
elif(UNIX)
    target_link_libraries(pool_app PRIVATE Threads::Threads)
endif()


# --- Unit Tests ---
add_executable(unit_tests
    tests/main_test.cpp
    tests/test_producer.cpp
    tests/test_consumer.cpp
    # Add other .cpp test files
)
target_compile_features(unit_tests PRIVATE cxx_std_20)
if(UNIX)
    target_link_libraries(unit_tests PRIVATE Threads::Threads gtest_main)
else() # e.g. Windows
    target_link_libraries(unit_tests PRIVATE gtest_main) # Threads::Threads might not be needed or named differently
endif()


# Include Directories
target_include_directories(pool_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add test to CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)

# Optional: Compiler flags
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    target_compile_options(pool_app PRIVATE -Wall -Wextra -Wpedantic) # -Werror
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()