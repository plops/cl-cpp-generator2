cmake_minimum_required(VERSION 3.10)

project(my_cpp_programs)

set(CMAKE_CXX_STANDARD 14)

# Add all cpp files in the current directory as executables
file(GLOB CPP_FILES *.cpp)

# Output the total number of cpp files found
message(STATUS "Found ${CPP_FILES} .cpp files.")

# Check if the list of files is empty
if(CPP_FILES)
  foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(EXECUTABLE_NAME ${CPP_FILE} NAME_WE)
    message(STATUS "Adding executable: ${EXECUTABLE_NAME}")
    add_executable(${EXECUTABLE_NAME} ${CPP_FILE})
  endforeach()
else()
  message(WARNING "No .cpp files found.")
endif()
