cmake_minimum_required(VERSION 3.6)
project(triangle)
set(CMAKE_CXX_STANDARD 26)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH ~/vulkan/)

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
            -O0
            #-Os
            #-O3
            #-ffunction-sections -fdata-sections # don't use together with flto
            #-g -g3 -ggdb -gdwarf-4   # useful if you want line numbers in *.nm output
    )
    #set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # flto


    #   -gc-sections: Enables "garbage collection" at the linker stage. The linker will analyze your code and discard any unused functions or data sections to reduce the final executable size.
    #   --print-gc-sections: Instructs the linker to generate a report detailing which sections were removed during the garbage collection process. This can help you pinpoint potential areas for further size optimization.
    #   --print-memory-usage: Tells the linker to provide a summary of the memory usage of your executable, broken down into categories like code, data, etc. This helps analyze memory consumption.
    #   -Map=...: Generates a map file that details the memory layout of your linked executable.
    #   --cref: Enables the generation of a cross-reference table. This table provides information about how symbols (functions, variables) are defined and used throughout your codebase.

    #add_link_options(-Wl,-gc-sections,--print-memory-usage,--print-gc-sections,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map,--cref)
endif ()


find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

find_package(glslang CONFIG REQUIRED)
find_package(Threads REQUIRED)

#find_package(x11)

include_directories(
        ~/src/Daxa/include
        ~/vulkan/include
)


add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        #src/main_tut.cpp
        /home/martin/src/Daxa/src/cpp_wrapper.cpp

        /home/martin/src/Daxa/src/impl_device.cpp
        /home/martin/src/Daxa/src/impl_features.cpp
        /home/martin/src/Daxa/src/impl_instance.cpp
        /home/martin/src/Daxa/src/impl_core.cpp
        /home/martin/src/Daxa/src/impl_pipeline.cpp
        /home/martin/src/Daxa/src/impl_swapchain.cpp
        /home/martin/src/Daxa/src/impl_command_recorder.cpp
        /home/martin/src/Daxa/src/impl_gpu_resources.cpp
        /home/martin/src/Daxa/src/impl_sync.cpp
        /home/martin/src/Daxa/src/impl_dependencies.cpp
        /home/martin/src/Daxa/src/impl_timeline_query.cpp

        /home/martin/src/Daxa/src/utils/impl_task_graph.cpp
        #/home/martin/src/Daxa/src/utils/impl_imgui.cpp
        #/home/martin/src/Daxa/src/utils/impl_fsr2.cpp
        /home/martin/src/Daxa/src/utils/impl_mem.cpp
        /home/martin/src/Daxa/src/utils/impl_pipeline_manager.cpp

)
set(HOME_PATH /home/martin)
target_compile_definitions(${PROJECT_NAME}
        PUBLIC
        DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_GLSLANG=true
        DAXA_BUILT_WITH_UTILS_TASK_GRAPH=true
        "DAXA_CMAKE_EXPORT="
        DAXA_BUILT_WITH_X11=true
        DAXA_BUILT_WITH_UTILS_MEM=true
        DAXA_INCLUDE="${HOME_PATH}/src/Daxa/include"
        SHADER_PATH="${CMAKE_SOURCE_DIR}/src"
)
#target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_link_libraries(
        ${PROJECT_NAME}
        # /home/martin/vulkan/lib64/libdaxa.a
        glfw
        #vulkan
        X11

        glslang::glslang
        #glslang::SPIRV
        GPUOpen::VulkanMemoryAllocator
        Vulkan::Vulkan
)

target_compile_options(${PROJECT_NAME}
        PRIVATE
        -fdiagnostics-color=always
)


