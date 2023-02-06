#include "c_resource.hpp"
#include "fatheader.hpp"
constexpr int SAMPLE_RATE = 44100;
constexpr int CHANNELS = 2;
constexpr int BUFFER_SIZE = 8192;
using MainLoop =
    stdex::c_resource<pw_main_loop, pw_main_loop_new, pw_main_loop_destroy>;

using Context =
    stdex::c_resource<pw_context, pw_context_new, pw_context_destroy>;

int main(int argc, char **argv) {
  pw_init(&argc, &argv);
  fmt::print("  pw_get_headers_version()='{}'  pw_get_library_version()='{}'\n",
             pw_get_headers_version(), pw_get_library_version());
  MainLoop main_loop(nullptr);
  Context context;
  context = {pw_main_loop_get_loop(main_loop), nullptr, 0};

  return 0;
}
