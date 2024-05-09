#define _REENTRANT
#include <iostream>
#include <spdlog/spdlog.h>
extern "C" {
#include <gio/gio.h>
#include <pipewire/pipewire.h>
};

int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  pw_init(nullptr, nullptr);
  {
    g_autoptr(GError) error = nullptr;
    auto *connection{g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &error)};
    if (error) {
      spdlog::info("d-bus connection failed  error->message='{}'",
                   error->message);
      return -1;
    }
    {
      auto *screencast_proxy{g_dbus_proxy_new_sync(
          connection, G_DBUS_PROXY_FLAGS_NONE, nullptr,
          "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop",
          "org.freedesktop.portal.ScreenCast", nullptr, &error)};
      if (error) {
        spdlog::info("d-bus proxy retrieval failed  error->message='{}'",
                     error->message);
        return -1;
      }
      auto cachedSourceTypes{g_dbus_proxy_get_cached_property(
          screencast_proxy, "AvailableSourceTypes")};
      if (cachedSourceTypes) {
        auto availableCaptureTypes{g_variant_get_uint32(cachedSourceTypes)};
        auto CAPTURE_TYPE_MONITOR{1 << 0};
        auto CAPTURE_TYPE_WINDOW{1 << 1};
        auto CAPTURE_TYPE_VIRTUAL{1 << 2};
        auto desktopCaptureAvailable{
            (0) != ((availableCaptureTypes) && (CAPTURE_TYPE_MONITOR))};
        auto windowCaptureAvailable{
            (0) != ((availableCaptureTypes) && (CAPTURE_TYPE_WINDOW))};
        spdlog::info(
            "  availableCaptureTypes='{}'  desktopCaptureAvailable='{}'  "
            "windowCaptureAvailable='{}'",
            availableCaptureTypes, desktopCaptureAvailable,
            windowCaptureAvailable);
        // init capture

        auto cancellable{g_cancellable_new()};
        auto connection{portal_get_dbus_connection()};
        if (!(connection)) {
          spdlog::info("can't get connection");
          return -1;
        }
        char *request_path, *request_token;
        portal_create_request_path(&request_path, &request_token);
        spdlog::info("  request_path='{}'  request_token='{}'", request_path,
                     request_token);
      }
    }
  }
}
