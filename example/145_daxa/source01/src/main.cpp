#include "VWindow.h"

#include <iostream>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

using namespace daxa::task_resource_uses;
using namespace daxa::types;

int main(int argc, char const *argv[])
{
    // Create a window
    auto window = VWindow("Learn Daxa", 860, 640);

    auto instance = daxa::create_instance({});
    auto device = instance.create_device({
    .selector = [](daxa::DeviceProperties const & device_props) -> daxa::i32
    {
        daxa::i32 score = 0;
        switch (device_props.device_type)
        {
        case daxa::DeviceType::DISCRETE_GPU: score += 10000; break;
        case daxa::DeviceType::VIRTUAL_GPU: score += 1000; break;
        case daxa::DeviceType::INTEGRATED_GPU: score += 100; break;
        default: break;
        }
        score += static_cast<daxa::i32>(device_props.limits.max_memory_allocation_count / 100000);
        return score;
    },
    .name = "my device",
});

    auto swapchain = device.create_swapchain({
    // this handle is given by the windowing API
    .native_window = window.get_native_handle(),
    // The platform would also be retrieved from the windowing API,
    // or by hard-coding it depending on the OS.
    .native_window_platform = window.get_native_platform(),
    // Here we can supply a user-defined surface format selection
    // function, to rate formats. If you don't care what format the
    // swapchain images are in, then you can just omit this argument
    // because it defaults to `daxa::default_format_score(...)`
    .surface_format_selector = [](daxa::Format format)
    {
        switch (format)
        {
        case daxa::Format::R8G8B8A8_UINT: return 100;
        default: return daxa::default_format_score(format);
        }
    },
    .present_mode = daxa::PresentMode::MAILBOX,
    .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
    .name = "my swapchain",
});

    auto swapchain_image = swapchain.acquire_next_image();
    // Daxa code goes here...
    while (!window.should_close())
    {
        window.update();
    }

    return 0;
}
