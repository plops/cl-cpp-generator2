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
    // Daxa code goes here...

    while (!window.should_close())
    {
        window.update();
    }

    return 0;
}
