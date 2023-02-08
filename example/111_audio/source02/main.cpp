#include "../source00/c_resource.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fmt/core.h>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
// https://docs.pipewire.org/tutorial4_8c-example.html
constexpr int DEFAULT_RATE = 44100;
constexpr int DEFAULT_CHANNELS = 2;
constexpr double DEFAULT_VOLUME = 0.7;
using MainLoop =
    stdex::c_resource<pw_main_loop, pw_main_loop_new, pw_main_loop_destroy>;

using Context =
    stdex::c_resource<pw_context, pw_context_new, pw_context_destroy>;

void local_pw_registry_destroy(pw_registry *registry) {
  fmt::print("local_pw_registry_destroy\n");
  pw_proxy_destroy(reinterpret_cast<pw_proxy *>(registry));
}
using Core = stdex::c_resource<pw_core, pw_context_connect, pw_core_disconnect>;

using Registry = stdex::c_resource<pw_registry, pw_core_get_registry,
                                   local_pw_registry_destroy>;

struct Data {
  pw_main_loop *loop;
  pw_stream *stream;
  double accumulator
};

void on_process(void *userdata) {
  fmt::print("on_process\n");
  auto data = reinterpret_cast<Data *>(userdata);
  auto b = pw_stream_dequeue_buffer(data->stream);
  if (nullptr == b) {
    fmt::print("out of buffers\n");
    return;
  }
  auto buf = b->buffer;
  auto dst = buf->datas[0].data;
  if (nullptr == dst) {
    return;
  }
  auto stride = (DEFAULT_CHANNELS * sizeof(int16_t));
  auto n_frames = ((buf->datas[0].maxsize) / (stride));
  for (auto i = 0; i < n_frames; i += 1) {
    data->accumulator += (((2764.60153515901800000000000000)) / (DEFAULT_RATE));
    if ((6.28318530717958600000000000000) <= data->accumulator) {
      (data->accumulator) -= ((6.28318530717958600000000000000));
    }
    val = (DEFAULT_VOLUME * sin(data->accumulator) * 16767);

    for (auto c = 0; c < DEFAULT_CHANNELS; c += 1) {
      *dst++ = val;
    }
  }
  buf->datas[0].chunk->offset = 0;

  buf->datas[0].chunk->stride = stride;

  buf->datas[0].chunk->size = (stride * n_frames);

  pw_stream_queue_buffer(data->stream, b);
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
      fmt::print("error: connection with pipewire daemon failed\n");
    }
    return v;
  })();
  auto registry = Registry(core, PW_VERSION_REGISTRY, 0);
  auto registry_listener = spa_hook();
  spa_zero(registry_listener);
  pw_stream_events stream_events = {.version = PW_VERSION_STREAM_EVENTS,
                                    .process = on_process};

  pw_registry_add_listener(reinterpret_cast<spa_interface *>(registry.get()),
                           &registry_listener, &registry_events, nullptr);
  roundtrip(core, main_loop);

  return 0;
}
