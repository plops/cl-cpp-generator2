//
// Created by martin on 2/7/25.
//

#include "window.h"
#include <daxa/daxa.hpp>
using namespace daxa;

int main(int argc, char const* argv[]) {
    // Create a window
    auto window = AppWindow("Learn Daxa", 860, 640);

    auto instance{create_instance({})};
    auto device{instance.create_device_2(instance.choose_device({},{}))};
    auto swapchain{
    device.create_swapchain(
        {.native_window{window.get_native_handle()},
        .native_window_platform{window.get_native_platform()},})};
    // Main loop
    while (!window.should_close()) {
        window.update();
    }

    return 0;
}