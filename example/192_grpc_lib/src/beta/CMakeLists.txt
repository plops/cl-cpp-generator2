add_library(fusion_lib STATIC
    SensorFusion.cpp
)
set_property(TARGET fusion_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(fusion_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

set(FUSION_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

add_executable(beta_server
    main.cpp
)

target_link_libraries(beta_server
    PRIVATE
        fusion_lib
        proto_lib
        gRPC::grpc++
)

target_include_directories(beta_server
    PRIVATE
        ${PROTO_LIB_INCLUDE_DIR}
)
