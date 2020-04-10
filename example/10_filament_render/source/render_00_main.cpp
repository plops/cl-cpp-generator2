
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cstdlib>
#include <filament/Engine.h>
#include <filament/FilamentAPI.h>

State state = {};
using namespace std::chrono_literals;
int main() {
  auto engine = filament::Engine::create();
  engine->destroy(&engine);
  return 0;
};