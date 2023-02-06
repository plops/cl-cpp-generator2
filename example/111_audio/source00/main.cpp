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

void registry_event_global(void *data, uint32_t id, uint32_t permissions,
                           const char *type, uint32_t version,
                           const struct spa_dict *props) {
  fmt::print("  id='{}'  type='{}'  version='{}'\n", id, type, version);
}

int main(int argc, char **argv) {
  pw_init(&argc, &argv);
  fmt::print("  pw_get_headers_version()='{}'  pw_get_library_version()='{}'\n",
             pw_get_headers_version(), pw_get_library_version());
  auto main_loop = MainLoop(nullptr);
  auto context = Context(pw_main_loop_get_loop(main_loop), nullptr, 0);
  auto core = ([&context]() {
    auto v = Core(context, nullptr, 0);
    if (nullptr == v) {
      fmt::print("error: core\n");
    }
    return v;
  })();
  auto registry = Registry(core, PW_VERSION_REGISTRY, 0);
  auto registry_listener = spa_hook();
  spa_zero(registry_listener);
  pw_registry_events registry_events = {.version = PW_VERSION_REGISTRY_EVENTS,
                                        .global = registry_event_global};

  pw_registry_add_listener(reinterpret_cast<spa_interface *>(registry.get()),
                           &registry_listener, &registry_events, nullptr);

  pw_main_loop_run(main_loop);
  return 0;
}
