#include <cstdlib>
#include <cstring>
#include <vector>
#include <xcb/xcb.h>

int main() {
  auto *conn = xcb_connect(nullptr, nullptr);
  if (xcb_connection_has_error(conn) != 0) {
    return 1;
  }
  auto *screen = xcb_setup_roots_iterator(xcb_get_setup(conn)).data;
  auto win = xcb_generate_id(conn);
  auto mask = ((XCB_CW_BACK_PIXEL) | (XCB_CW_EVENT_MASK));
  auto values = std::vector<uint32_t>(
      {screen->white_pixel,
       ((XCB_EVENT_MASK_EXPOSURE) | (XCB_EVENT_MASK_KEY_PRESS) |
        (XCB_EVENT_MASK_BUTTON_PRESS))});
  xcb_create_window(conn, XCB_COPY_FROM_PARENT, win, screen->root, 0, 0, 600,
                    400, 2, XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual,
                    mask, values.data());
  auto gc = xcb_generate_id(conn);
  auto font = xcb_generate_id(conn);
  const auto *fontName = "-*-terminal-medium-*-*-*-14-*-*-*-*-*-iso8859-*";
  auto fontMask = ((XCB_GC_FOREGROUND) | (XCB_GC_BACKGROUND) | (XCB_GC_FONT));
  auto fontValues =
      std::vector<uint32_t>({screen->black_pixel, screen->white_pixel, font});
  xcb_open_font(conn, font, strlen(fontName), fontName);
  xcb_create_gc(conn, gc, win, fontMask, fontValues.data());
  xcb_map_window(conn, win);
  xcb_flush(conn);

  auto done = true;
  const auto *helloString = "Hello";
  while (done) {
    auto *event = xcb_wait_for_event(conn);
    if ((event) == nullptr) {
      break;
    }
    switch (event->response_type) {
    case XCB_EXPOSE: {
      xcb_clear_area(conn, 0, win, 0, 0, 0, 0);
      xcb_image_text_8(conn, strlen(helloString), win, gc, 50, 50, helloString);
      xcb_flush(conn);
      break;
    }
    case XCB_MAPPING_NOTIFY: {
      break;
    }
    case XCB_BUTTON_PRESS: {
      auto *ev = reinterpret_cast<xcb_button_press_event_t *>(event);
      auto x = ev->event_x;
      auto y = ev->event_y;
      xcb_flush(conn);

      break;
    }
    case XCB_KEY_PRESS: {
      auto geom_c = xcb_get_geometry(conn, win);
      auto *geom = xcb_get_geometry_reply(conn, geom_c, nullptr);
      xcb_change_gc(conn, gc, XCB_GC_FOREGROUND, &screen->white_pixel);
      const char *s = "Hello World";
      xcb_image_text_8(conn, strlen(s), win, gc, geom->x, geom->y, s);
      xcb_flush(conn);
      free(geom);

      break;
    }
    }
    free(event);
  }
  xcb_disconnect(conn);
  return 0;
}
