cmake_minimum_required(VERSION 2.9)
project(NANOPB_CMAKE_SIMPLE)

# install dependencies:
# cd /home/martin/src; git clone https://github.com/nanopb/nanopb
# pip install protobuf grpcio-tools

set(CMAKE_MODULE_PATH /home/martin/src/nanopb/extra/)
find_package(Nanopb REQUIRED)
include_directories(${NANOPB_INCLUDE_DIRS})

nanopb_generate_cpp(PROTO_SRCS PROTO_HDRS data.proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g -O0")
set(SERVER_SOURCE_FILES
 /home/martin/stage/cl-cpp-generator2/example/121_nanopb/source01/main.cpp
)
add_executable(server
 ${SERVER_SOURCE_FILES}
 ${PROTO_SRCS}
 ${PROTO_HDRS}
 )
