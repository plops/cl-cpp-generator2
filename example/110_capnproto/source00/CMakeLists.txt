
cmake_minimum_required(VERSION 3.12)
project(my_service)

set(CMAKE_CXX_STANDARD 17)

find_package(CapnProto CONFIG REQUIRED)

add_executable(my_service
  main.cpp
  my_service.capnp.c++
)

target_link_libraries(my_service
  CapnProto::capnp-rpc
  CapnProto::capnp
)

target_include_directories(my_service PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/my_service.capnp.c++
  COMMAND capnp compile -oc++ ${CMAKE_CURRENT_SOURCE_DIR}/my_service.capnp -I${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS my_service.capnp
)

add_custom_target(my_service_capnp DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/my_service.capnp.c++)

add_dependencies(my_service my_service_capnp)
