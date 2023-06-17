cmake_minimum_required(VERSION 3.10)

project(ParticleInABox)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Armadillo library directory
link_directories(/home/martin/src/armadillo-code/b/)

# Add include directory
# Assuming that Armadillo headers are in the 'include' subdirectory of armadillo-code
include_directories(/home/martin/src/armadillo-code/include/)

# Add executable
add_executable(ParticleInABox main.cpp)

# Link Armadillo
target_link_libraries(ParticleInABox armadillo)
