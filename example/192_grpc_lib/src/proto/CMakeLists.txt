# Find the protoc and grpc_cpp_plugin executables
find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# Define paths for generated files
set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/processor.pb.cc")
set(PROTO_HDR "${CMAKE_CURRENT_BINARY_DIR}/processor.pb.h")
set(GRPC_SRC "${CMAKE_CURRENT_BINARY_DIR}/processor.grpc.pb.cc")
set(GRPC_HDR "${CMAKE_CURRENT_BINARY_DIR}/processor.grpc.pb.h")

# Custom command to generate the files
add_custom_command(
    OUTPUT "${PROTO_SRC}" "${PROTO_HDR}" "${GRPC_SRC}" "${GRPC_HDR}"
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         -I "${CMAKE_CURRENT_SOURCE_DIR}"
         "${CMAKE_CURRENT_SOURCE_DIR}/processor.proto"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/processor.proto"
    COMMENT "Generating gRPC/Protobuf files from processor.proto"
    VERBATIM
)

# Create the library
add_library(proto_lib STATIC
    ${PROTO_SRC}
    ${GRPC_SRC}
)
set_property(TARGET proto_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(proto_lib
    PUBLIC
        gRPC::grpc++
        protobuf::libprotobuf
)

target_include_directories(proto_lib
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

# Ensure the generated files are ready before building the library
set_source_files_properties(${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR} PROPERTIES GENERATED TRUE)

# Export variables for other targets
set(PROTO_LIB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
