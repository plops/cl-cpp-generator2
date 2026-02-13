# Build Local Variant
add_executable(alpha_local
    main.cpp
)

target_link_libraries(alpha_local
    PRIVATE
        libone
)

# Build Remote Variant
add_executable(alpha_remote
    main.cpp
)

target_link_libraries(alpha_remote
    PRIVATE
        libtwo
)

# Common includes
target_include_directories(alpha_local PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(alpha_remote PRIVATE ${PROJECT_SOURCE_DIR}/include)
