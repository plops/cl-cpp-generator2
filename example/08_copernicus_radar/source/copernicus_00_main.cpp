
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
State state = {};
int main() {
  state._start_time = now();
  run();
};