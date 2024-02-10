#include "Window.h"

#include <iostream>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

using namespace daxa::task_resource_uses;
using namespace daxa::types;

int main(int argc, char const *argv[])
{
    // Create a window
    auto window = Window("Learn Daxa", 860, 640);

    // Daxa code goes here...

    while (!window.should_close())
    {
        window.update();
    }

    return 0;
}
