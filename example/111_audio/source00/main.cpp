#include "c_resource.hpp"
#include "fatheader.hpp"
constexpr int SAMPLE_RATE = 44100;
constexpr int CHANNELS = 2;
constexpr int BUFFER_SIZE = 8192;
using MainLoop =
    stdex::c_resource<pw_main_loop, pw_main_loop_new, pw_main_loop_destroy>;

using Context =
    stdex::c_resource<pw_context, pw_context_new, pw_context_destroy>;

void local_pw_registry_destroy(pw_registry *registry) {
  pw_proxy_destroy(reinterpret_cast<pw_proxy *>(registry));
}
using Core = stdex::c_resource<pw_core, pw_context_connect, pw_core_disconnect>;

using Registry = stdex::c_resource<pw_registry, pw_core_get_registry,
                                   local_pw_registry_destroy>;

int main(int argc, char **argv) {
  pw_init(&argc, &argv);
  fmt::print("  pw_get_headers_version()='{}'  pw_get_library_version()='{}'\n",
             pw_get_headers_version(), pw_get_library_version());
  auto main_loop = MainLoop(nullptr);
  auto context = Context(pw_main_loop_get_loop(main_loop), nullptr, 0);
  auto core = Core(context, nullptr, 0);
  auto registry = Registry(core, PW_VERSION_REGISTRY, 0);

  return 0;
}
