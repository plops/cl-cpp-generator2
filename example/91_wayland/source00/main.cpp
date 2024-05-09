#include <iostream>
#include <spdlog/spdlog.h>
extern "C" {
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <wayland-client.h>
};
struct wl_compositor *wl_compositor = nullptr;
struct wl_shm *wl_shm = nullptr;
struct wl_output *wl_output = nullptr;

void registry_handle_global(void *data, struct wl_registry *registry,
                            uint32_t id, const char *interface,
                            uint32_t version) {
  spdlog::info("  id='{}'  version='{}'  interface='{}'", id, version,
               interface);
  if (("wl_compositor") == (std::string_view(interface))) {
    spdlog::info("wl_compositor  id='{}'  version='{}'  interface='{}'", id,
                 version, interface);
    (wl_compositor) = (static_cast<struct wl_compositor *>(
        wl_registry_bind(registry, id, &wl_compositor_interface, version)));
    return;
  }
  if (("wl_shm") == (std::string_view(interface))) {
    spdlog::info("wl_shm  id='{}'  version='{}'  interface='{}'", id, version,
                 interface);
    (wl_shm) = (static_cast<struct wl_shm *>(
        wl_registry_bind(registry, id, &wl_shm_interface, version)));
    return;
  }
  if (("wl_output") == (std::string_view(interface))) {
    spdlog::info("wl_output  id='{}'  version='{}'  interface='{}'", id,
                 version, interface);
    (wl_output) = (static_cast<struct wl_output *>(
        wl_registry_bind(registry, id, &wl_output_interface, version)));
    return;
  }
}

void registry_handle_global_remove(void *data, struct wl_registry *registry,
                                   uint32_t id) {
  if ((wl_compositor) &
      ((id) ==
       (wl_proxy_get_id(reinterpret_cast<struct wl_proxy *>(wl_compositor))))) {
    spdlog::info("wl_compositor  id='{}'", id);
    (wl_compositor) = (nullptr);
  }
  if ((wl_shm) & ((id) == (wl_proxy_get_id(
                              reinterpret_cast<struct wl_proxy *>(wl_shm))))) {
    spdlog::info("wl_shm  id='{}'", id);
    (wl_shm) = (nullptr);
  }
  if ((wl_output) &
      ((id) ==
       (wl_proxy_get_id(reinterpret_cast<struct wl_proxy *>(wl_output))))) {
    spdlog::info("wl_output  id='{}'", id);
    (wl_output) = (nullptr);
  }
}

(static const struct wl_registry_listener registry_listener) =
    ({registry_handle_global, registry_handle_global_remove});

int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  {
    auto *display{wl_display_connect("wayland-0")};
    if ((nullptr) == (display)) {
      spdlog::info("can't connect to display");
      return -1;
    }
    auto *registry{wl_display_get_registry(display)};
    spdlog::info("add listener..");
    wl_registry_add_listener(registry, &registry_listener, nullptr);
    spdlog::info("roundtrip..");
    wl_display_roundtrip(display);
    spdlog::info("dispatch..");
    wl_display_dispatch(display);
    if (!(wl_compositor)) {
      spdlog::info("missing wl_compositor");
      return -1;
    }
    if (!(wl_shm)) {
      spdlog::info("missing wl_shm");
      return -1;
    }
    if (!(wl_output)) {
      spdlog::info("missing wl_output");
      return -1;
    }
    {
      auto shm_fn{"/my-wayland-pool"};
      auto fd{shm_open(shm_fn, (O_RDWR) || (O_CREAT) || (O_EXCL),
                       (S_IRUSR) || (S_IWUSR))};
      auto width{1920};
      auto height{1080};
      auto stride{4};
      auto format{WL_SHM_FORMAT_ARGB8888};
      auto size{(width) * (height) * (stride)};
      if ((fd) < (0)) {
        spdlog::info(
            "shm_open failed.  shm_fn='{}'  errno='{}'  strerror(errno)='{}'",
            shm_fn, errno, strerror(errno));
        shm_unlink(shm_fn);
        return -1;
      }
      if ((ftruncate(fd, size)) < (0)) {
        spdlog::info("ftruncate failed");
        return -1;
      }
    }
    spdlog::info("disconnect..");
    wl_display_disconnect(display);
  }
}
