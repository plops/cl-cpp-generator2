#include "fatheader.hpp"
constexpr int SAMPLE_RATE = 44100;
constexpr int CHANNELS = 2;
constexpr int BUFFER_SIZE = 8192;

int main(int argc, char **argv) {
  spa_handle_factory_enum(&factory, SPA_TYPE_INTERFACE_Node, 0, 0);
  return 0;
}
