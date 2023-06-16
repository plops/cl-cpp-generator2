cmake_minimum_required(VERSION 3.10)

project(my_cpp_programs)

set(CMAKE_CXX_STANDARD 14)

# Add all cpp files in the current directory as executables
file(GLOB CPP_FILES *.cpp)
foreach(CPP_FILE ${CPP_FILES})
  get_filename_component(EXECUTABLE_NAME ${CPP_FILE} NAME_WE)
  add_executable(${EXECUTABLE_NAME} ${CPP_FILE})
endforeach()
