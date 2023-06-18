# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(ArucoMarkerDetection)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify the location of the local opencv build
set(OpenCV_DIR "/home/martin/opencv/lib/cmake/opencv4")

# find OpenCV
find_package(OpenCV REQUIRED)

# specify the source files
set(SOURCE_FILES
    main.cpp
)

# specify the executable
add_executable(ArucoMarkerDetection ${SOURCE_FILES})

# specify the libraries to link with the executable
target_link_libraries(ArucoMarkerDetection ${OpenCV_LIBS})
