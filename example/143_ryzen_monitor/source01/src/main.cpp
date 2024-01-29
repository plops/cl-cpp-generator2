#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <format>

void glfw_error_callback(int err, const char *description) {
  std::cout << std::format(" errror='{}' description='{}'\n", errror,
                           description);
}

int main(int argc, char **argv) {}
