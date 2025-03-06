cmake_minimum_required(VERSION 3.6)
project(triangle)
set(CMAKE_CXX_STANDARD 26)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH ~/vulkan/)


# AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# UndefinedBehaviorSanitizer
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if (ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif ()

# ThreadSanitizer
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif ()

# LeakSanitizer
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
if (ENABLE_LSAN)
    add_compile_options(-fsanitize=leak)
    add_link_options(-fsanitize=leak)
endif ()



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


find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

find_package(glslang CONFIG REQUIRED)
find_package(Threads REQUIRED)

#find_package(x11)

include_directories(
        ~/src/Daxa/include
        ~/vulkan/include
        ~/src/imgui
        ~/src/implot
)

set(DEP_SRC_PATH /home/martin/src)

add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        #src/main_tut.cpp
        ${DEP_SRC_PATH}/Daxa/src/cpp_wrapper.cpp

        ${DEP_SRC_PATH}/Daxa/src/impl_device.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_features.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_instance.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_core.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_pipeline.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_swapchain.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_command_recorder.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_gpu_resources.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_sync.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_dependencies.cpp
        ${DEP_SRC_PATH}/Daxa/src/impl_timeline_query.cpp

        ${DEP_SRC_PATH}/Daxa/src/utils/impl_task_graph.cpp
        ${DEP_SRC_PATH}/Daxa/src/utils/impl_imgui.cpp
        #${DEP_SRC_PATH}/Daxa/src/utils/impl_fsr2.cpp
        ${DEP_SRC_PATH}/Daxa/src/utils/impl_mem.cpp
        ${DEP_SRC_PATH}/Daxa/src/utils/impl_pipeline_manager.cpp

        ${DEP_SRC_PATH}/imgui/imgui.cpp
        ${DEP_SRC_PATH}/imgui/imgui_demo.cpp
        ${DEP_SRC_PATH}/imgui/imgui_draw.cpp
        ${DEP_SRC_PATH}/imgui/imgui_tables.cpp
        ${DEP_SRC_PATH}/imgui/imgui_widgets.cpp
        ${DEP_SRC_PATH}/imgui/backends/imgui_impl_glfw.cpp
        # ${DEP_SRC_PATH}/imgui/backends/imgui_impl_vulkan.cpp

        ${DEP_SRC_PATH}/implot/implot.cpp
        ${DEP_SRC_PATH}/implot/implot_demo.cpp
        ${DEP_SRC_PATH}/implot/implot_items.cpp


)
set(HOME_PATH /home/martin)
target_compile_definitions(${PROJECT_NAME}
        PUBLIC
        DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_GLSLANG=true
        DAXA_BUILT_WITH_UTILS_TASK_GRAPH=true
        "DAXA_CMAKE_EXPORT="
        DAXA_BUILT_WITH_X11=true
        DAXA_BUILT_WITH_UTILS_MEM=true
        DAXA_BUILT_WITH_UTILS_IMGUI=true
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


