add_library(libone SHARED
    LocalProcessor.cpp
)

target_link_libraries(libone
    PRIVATE
        fusion_lib
)

target_include_directories(libone
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${FUSION_LIB_INCLUDE_DIR}
)

# Export its name so Alpha can link against it
set(LIBONE_TARGET libone PARENT_SCOPE)
