//
// Created by martin on 2/7/25.
//

#include "window.h"
#include <daxa/daxa.hpp>

int main(int argc, char const* argv[]) {
    // Create a window
    auto window = AppWindow("Learn Daxa", 860, 640);

    auto instance{daxa::create_instance({})};

    // Main loop
    while (!window.should_close()) {
        window.update();
    }

    return 0;
}