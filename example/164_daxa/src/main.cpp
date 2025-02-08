//
// Created by martin on 2/7/25.
//

/**
 * needs ~/src/Daxa and ~/vulkan VulkanMemory...
 */

#include <array>
#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>
#include "shared.inl"
#include "window.h"
using namespace daxa;


void upload_vertex_data_task(TaskGraph& tg, TaskBufferView vertices)
{
    tg.add_task({.attachments{{inl_attachment(TaskBufferAccess::TRANSFER_WRITE, vertices)}},
                 .task{[=](TaskInterface ti)
                       {
                           constexpr float n{-.5f}, p{.5f}, z{.0f}, o{1.f};
                           auto data{std::array{
                               MyVertex{.position{n, p, z}, .color{o, z, z}},
                               MyVertex{.position{p, p, z}, .color{z, o, z}},
                               MyVertex{.position{z, n, z}, .color{z, z, o}},
                           }};
                           auto staging_buffer_id
                           {
                               ti.device.create_buffer({.size{sizeof(data)},
                                                        .allocate_info{MemoryFlagBits::HOST_ACCESS_RANDOM},
                                                        .name{"my_staging_buffer"}});
                           };
                       }}});
}

int main(int argc, char const* argv[])
{
    // Create a window
    auto window = AppWindow("Learn Daxa", 860, 640);

    auto instance{create_instance({})};
    auto device{instance.create_device_2(instance.choose_device({}, {}))};
    auto swapchain{device.create_swapchain({.native_window{window.get_native_handle()},
                                            .native_window_platform{AppWindow::get_native_platform()},
                                            .surface_format_selector{[](Format format)
                                                                     {
                                                                         switch (format)
                                                                         {
                                                                         case Format::R8G8B8A8_UINT:
                                                                             return 100;
                                                                         default:
                                                                             return default_format_score(format);
                                                                         }
                                                                     }},
                                            .present_mode{PresentMode::MAILBOX},
                                            .image_usage{ImageUsageFlagBits::TRANSFER_DST},
                                            .name{"my swapchain"}})};
    auto swapchain_image{swapchain.acquire_next_image()};

    auto pipeline_manager{PipelineManager(
        {.device{device},
         .shader_compile_options{.root_paths{"."}, .language{ShaderLanguage::GLSL}, .enable_debug_info{true}},
         .name{"my pipelinemanager"}})};

    std::shared_ptr<RasterPipeline> pipeline;
    {
        auto result = pipeline_manager.add_raster_pipeline({
            .vertex_shader_info{ShaderCompileInfo{.source{ShaderFile{"main.glsl"}}}},
            .fragment_shader_info{ShaderCompileInfo{.source{ShaderFile{"main.glsl"}}}},
            .color_attachments{{.format{swapchain.get_format()}}},
            .raster{{}},
            .push_constant_size{sizeof(MyPushConstant)},
            .name{"my pipeline"},
        });
        if (result.is_err())
        {
            std::cerr << result.message() << std::endl;
            return -1;
        }
        pipeline = result.value();
    }

    auto buffer_id{device.create_buffer({.size{3 * sizeof(MyVertex)}, .name{"my vertex data"}})};
    // Main loop
    while (!window.should_close())
    {
        window.update();
    }

    return 0;
}
