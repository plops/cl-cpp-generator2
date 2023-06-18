# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(OpenCLCheck)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the 'cmake' directory to the CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# find_package(OpenCV REQUIRED)

# specify the location of the local opencv build
set(OpenCV_DIR "/home/martin/opencv")


# specify the location of the fruit library
set(FRUIT_INCLUDE_DIR "/home/martin/fruit/include")
set(FRUIT_LIBRARY "/home/martin/fruit/lib/libfruit.so")

#message(STATUS "Trying to find Fruit library in: ${FRUIT_LIBRARY}")
#message(STATUS "Trying to find Fruit headers in: ${FRUIT_INCLUDE_DIR}")

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
#find_package(Fruit REQUIRED)



include_directories(
 ${OpenCV_INCLUDE_DIRS}
 ${FRUIT_INCLUDE_DIR}
 # /home/martin/opencv/include/opencv4/
)

add_executable(OpenCLCheck
  main.cpp
  ArucoCheckerboardDisplay.cpp
)

target_link_libraries(OpenCLCheck
  ${OpenCV_LIBS}
  ${FRUIT_LIBRARY}
)
