add_library(proto_lib STATIC
    processor.proto
)

target_link_libraries(proto_lib
    PUBLIC
        gRPC::grpc++
        protobuf::libprotobuf
)

target_include_directories(proto_lib
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

protobuf_generate(
    TARGET proto_lib
    LANGUAGE cpp
)

grpc_generate(
    TARGET proto_lib
    LANGUAGE cpp
)

# Export variables for other targets
set(PROTO_LIB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
