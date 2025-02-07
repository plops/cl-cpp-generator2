//
// Created by martin on 2/7/25.
//

#include "window.h"

int main(int argc, char const* argv[]) {
    // Create a window
    auto window = AppWindow("Learn Daxa", 860, 640);

    // Daxa rendering initialization code goes here...

    // Main loop
    while (!window.should_close()) {
        window.update();
    }

    return 0;
}