cmake_minimum_required(VERSION 3.20)
project(CppLuaImageDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(target_set_warnings target_name)
    set(CLANG_WARNINGS
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wcast-align
        -Wunused -Woverloaded-virtual -Wconversion
        -Wsign-conversion -Wdouble-promotion
        -Wformat=2 -Wimplicit-fallthrough
    )

    set(GCC_WARNINGS
        ${CLANG_WARNINGS}
        -Wmisleading-indentation -Wduplicated-cond
        -Wduplicated-branches -Wlogical-op
        -Wuseless-cast
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PRIVATE ${CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${target_name} PRIVATE ${GCC_WARNINGS})
    endif()

    if(WARNINGS_AS_ERRORS)
        target_compile_options(${target_name} PRIVATE -Werror)
    endif()
endfunction()

# Add options for ublasan (undefined, leak and asan enabled)
option(ENABLE_UBLASAN "Enable undefined leak and address sanitizers" OFF)
# Option to enable march=native
option(ENABLE_MARCH_NATIVE "Enable march=native for optimization" ON)
option(DISABLE_RTTI "Disable RTTI for smaller binary" OFF)
option(DISABLE_EXCEPTIONS "Disable C++ Exceptions" OFF)

# Enable link time optimization (interprocedural optimization)
set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)


# Option to enable warnings as error
if(ENABLE_UBLASAN)
    # enable 3 sanitizers
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fsanitize=leak -fsanitize=address")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=undefined -fsanitize=leak -fsanitize=address")
endif()

if(ENABLE_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()
# Flags für Resource-Constrained Env (Global oder Target-Specific)
if(DISABLE_RTTI)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)
endif()

if(DISABLE_EXCEPTIONS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables>)
endif()


find_package(Lua REQUIRED)

add_executable(cpp_lua_image_demo
    src/main.cpp
)
target_set_warnings(cpp_lua_image_demo)

#Using SYSTEM is the standard way to say “this third-party header is trusted; don’t spam warnings”.
target_include_directories(cpp_lua_image_demo
        SYSTEM PRIVATE
        external/sol2/include
)

target_include_directories(cpp_lua_image_demo
        PRIVATE
        ${LUA_INCLUDE_DIR}
)

target_link_libraries(cpp_lua_image_demo PRIVATE
    ${LUA_LIBRARIES}
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(cpp_lua_image_demo PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(cpp_lua_image_demo PRIVATE -Wl,--gc-sections)
endif()
