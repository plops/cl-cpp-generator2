cmake_minimum_required(VERSION 3.16)
project(ModernCppVariant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads)
find_package(GTest)
# --- Google Test ---
#include(FetchContent)
#FetchContent_Declare(
#  googletest
#  GIT_REPOSITORY https://github.com/google/googletest.git
#  GIT_TAG       release-1.14.0 # Or a specific commit
#)
#FetchContent_MakeAvailable(googletest)

# --- Application ---
add_executable(variant
    src/variant.cpp
    # Add other .cpp files if they exist (most are header-only in this example)
)
target_compile_features(variant PRIVATE cxx_std_20)
if(WIN32)
    # Link against Ws2_32.lib for htons/ntohs on Windows if not using a cross-platform lib
    # For MinGW, it might be linked automatically or need -lws2_32
    # For MSVC, this is typically needed if <winsock2.h> is used directly.
    # However, arpa/inet.h is POSIX. On Windows, you might need a different approach or a library.
    # For simplicity, assuming POSIX or MinGW where arpa/inet.h might pull ws2_32.
elif(UNIX)
    target_link_libraries(variant PRIVATE Threads::Threads)
endif()


# Optional: Compiler flags
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    target_compile_options(variant PRIVATE -Wall -Wextra -Wpedantic) # -Werror
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()