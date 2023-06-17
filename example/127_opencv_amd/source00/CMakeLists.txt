# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(OpenCLCheck)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)



# specify the location of the fruit library
set(FRUIT_INCLUDE_DIR "/home/martin/src/fruit-3.7.1/include")
set(FRUIT_LIBRARY "/home/martin/src/fruit-3.7.1/b/src/libfruit.so")

include_directories(
 ${OpenCV_INCLUDE_DIRS}
 ${FRUIT_INCLUDE_DIR} )

add_executable(OpenCLCheck
  main.cpp
  ArucoCheckerboardDisplay.cpp
)

target_link_libraries(OpenCLCheck
  ${OpenCV_LIBS} # needs vtk and hdf5
  ${FRUIT_LIBRARY}
)
