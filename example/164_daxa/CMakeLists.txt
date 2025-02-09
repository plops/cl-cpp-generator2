cmake_minimum_required(VERSION 3.6)
project(triangle)
set(CMAKE_CXX_STANDARD 23)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG ON)
set(CMAKE_PREFIX_PATH ~/vulkan/)
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
target_compile_definitions(${PROJECT_NAME}
        PUBLIC
        DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_GLSLANG=true
        DAXA_BUILT_WITH_UTILS_TASK_GRAPH=true
        "DAXA_CMAKE_EXPORT="
        DAXA_BUILT_WITH_X11=true
        DAXA_BUILT_WITH_UTILS_MEM=true

)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_link_libraries(
        ${PROJECT_NAME}
        # /home/martin/vulkan/lib64/libdaxa.a
        glfw
        #vulkan
        X11

        glslang::glslang
        glslang::SPIRV
        GPUOpen::VulkanMemoryAllocator
        Vulkan::Vulkan
)
