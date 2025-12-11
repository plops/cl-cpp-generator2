cmake_minimum_required(VERSION 3.10)

# Projektname. Pass das an, mir egal.
project(NiosLuaEmbedded C CXX)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Standards setzen. C++17 ist okay, solange der Nios-GCC nicht von 2010 ist.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

# Add options for ublasan (undefined, leak and asan enabled)
option(ENABLE_UBLASAN "Enable undefined leak and address sanitizers" ON)
# Option to enable march=native
option(ENABLE_MARCH_NATIVE "Enable march=native for optimization" ON)


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

# Pfad zu den Lua-Quellen basierend auf deiner Struktur
set(LUA_SRC_DIR "src/lua-5.4.8/src")

# ---------------------------------------------------------------------------
# Lua Library Build
# ---------------------------------------------------------------------------

# Alle .c Dateien holen
file(GLOB LUA_SOURCES "${LUA_SRC_DIR}/*.c")

# WICHTIG: Entferne lua.c und luac.c!
# Das sind die Command-Line-Tools (Interpreter und Compiler).
# Du willst Lua einbetten, nicht als OS-Tool bauen.
list(REMOVE_ITEM LUA_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/${LUA_SRC_DIR}/lua.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/${LUA_SRC_DIR}/luac.c"
)

# Erstelle eine statische Library. Shared Libs auf Nios II sind Schmerz.
#add_library(lua_lib STATIC ${LUA_SOURCES})

# Include-Pfade exportieren, damit main.cpp sie findet
#target_include_directories(lua_lib PUBLIC "${LUA_SRC_DIR}")
include_directories("${LUA_SRC_DIR}")

# Compiler-Flags für Lua
# LUA_32BITS: Zwingend für Nios II.
# LUA_USE_C89: Macht es kompatibler mit ranzigen Embedded-Compilern.
#target_compile_definitions(lua_lib PUBLIC
#    LUA_32BITS
#    LUA_USE_C89
#)

# Falls du size-optimization willst (empfohlen):
#if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
#    target_compile_options(lua_lib PRIVATE -Os -g0)
#endif()

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

add_executable(main_firmware src/main.cpp ${LUA_SOURCES})

# Linken
#target_link_libraries(main_firmware PRIVATE lua_lib)

# Linker-Flags für Embedded (GC Sections), falls nicht eh in deiner Toolchain:
# Entfernt ungenutzte Lua-Funktionen aus dem Binary.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(main_firmware PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(main_firmware PRIVATE -Wl,--gc-sections)
endif()
