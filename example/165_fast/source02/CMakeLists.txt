cmake_minimum_required(VERSION 3.31)
project(fast)
set(CMAKE_CXX_STANDARD 26)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_PREFIX_PATH /home/martin/vulkan)

find_package(benchmark)
find_package(Boost COMPONENTS container)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
            -Og
    )
    add_link_options(-Wl,--print-memory-usage,--print-gc-sections,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map,--cref)

    #    set(CPPCHECK_TEMPLATE "gcc")
    #    find_program(CPPCHECK cppcheck)
    #    find_program(CLANG_TIDY clang-tidy)
    #    if(CPPCHECK)
    #        set(CMAKE_CXX_CPPCHECK
    #                ${CPPCHECK}
    #                --template=${CPPCHECK_TEMPLATE}
    #                --enable=style,performance,warning,portability
    #                --inline-suppr
    #                --suppress=cppcheckError
    #                --suppress=internalAstError
    #                --suppress=unmatchedSuppression
    #                --suppress=preprocessorErrorDirective
    #                --suppress=exceptThrowInDestructor
    #                --suppress=functionStatic
    #                --inconclusive)
    #    endif()
    #    if(CLANG_TIDY)
    #        set(CMAKE_CXX_CLANG_TIDY
    #                ${CLANG_TIDY}
    #                --fix)
    #    endif()

else ()

    #set(CMAKE_UNITY_BUILD true) # merge c++ files
    add_compile_options(
            #-O0
            #-Os
            -O3 -march=native
            -ffunction-sections -fdata-sections # don't use together with flto
            #-g -g3 -ggdb -gdwarf-4   # useful if you want line numbers in *.nm output
    )
    #Cset(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # flto


    #   -gc-sections: Enables "garbage collection" at the linker stage. The linker will analyze your code and discard any unused functions or data sections to reduce the final executable size.
    #   --print-gc-sections: Instructs the linker to generate a report detailing which sections were removed during the garbage collection process. This can help you pinpoint potential areas for further size optimization.
    #   --print-memory-usage: Tells the linker to provide a summary of the memory usage of your executable, broken down into categories like code, data, etc. This helps analyze memory consumption.
    #   -Map=...: Generates a map file that details the memory layout of your linked executable.
    #   --cref: Enables the generation of a cross-reference table. This table provides information about how symbols (functions, variables) are defined and used throughout your codebase.

    add_link_options(-Wl,-gc-sections,--print-memory-usage,--print-gc-sections,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map,--cref)
endif ()


add_executable(
        ${PROJECT_NAME}
        src/main.cpp
)

include_directories(/home/martin/src/stable_vector)

target_link_libraries(
        ${PROJECT_NAME}
        #Boost::container
        benchmark::benchmark
)

target_compile_options(${PROJECT_NAME}
        PRIVATE
        -fdiagnostics-color=always
)


