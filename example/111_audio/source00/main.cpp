#include "fatheader.hpp"
constexpr int SAMPLE_RATE = 44100;
constexpr int CHANNELS = 2;
constexpr int BUFFER_SIZE = 8192;
struct data {
  bal loop;
};
typedef struct data data;

int main(int argc, char **argv) {
  pw_init(&argc, &argv);
  fmt::print("  pw_get_headers_version()='{}'  pw_get_library_version()='{}'\n",
             pw_get_headers_version(), pw_get_library_version());
  return 0;
}
