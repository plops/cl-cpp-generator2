//
// Created by martin on 2/7/25.
//

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include "window.h"
using namespace daxa;

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

    // requires DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG
    // auto pipeline_manager
    // {
    //     PipelineManager(
    //         {.device{device},
    //          .shader_compile_options{.root_paths{{"."}}.language{ShaderLanguage::GLSL}.enable_debug_info{true}},
    //          .name{"my pipelinemanager"}})
    // }
    // Main loop
    while (!window.should_close())
    {
        window.update();
    }

    return 0;
}
