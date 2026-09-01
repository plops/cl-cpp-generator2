cmake_minimum_required(VERSION 3.14)
project(vermell VERSION 1.0.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(server
 src/main.cpp
)
target_link_libraries(server vermell)
