add_library(libtwo SHARED
    RemoteProcessor.cpp
)

target_link_libraries(libtwo
    PRIVATE
        proto_lib
        gRPC::grpc++
)

target_include_directories(libtwo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTO_LIB_INCLUDE_DIR}
)

# Export its name so Alpha can link against it
set(LIBTWO_TARGET libtwo PARENT_SCOPE)
