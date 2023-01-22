#include <cstring>
#include <iostream>
#include <xcb/xcb.h>

int main() {
  auto *conn = xcb_connect(nullptr, nullptr);
  if (xcb_connection_has_error(conn) != 0) {
    return 1;
  }
  auto *screen = xcb_setup_roots_iterator(xcb_get_setup(conn)).data;
  auto win = xcb_generate_id(conn);
  xcb_create_window(conn, XCB_COPY_FROM_PARENT, win, screen->root, 0, 0, 600,
                    400, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual,
                    0, nullptr);
  xcb_gcontext_t gc = xcb_generate_id(conn);
  xcb_create_gc(conn, gc, win, XCB_GC_FOREGROUND, &screen->black_pixel);
  xcb_map_window(conn, win);
  xcb_flush(conn);

  while (true) {
    auto *event = xcb_wait_for_event(conn);
    if ((event) == nullptr) {
      break;
    }
    switch (((event->response_type) & (~(0x80)))) {
    case XCB_EXPOSE: {
      xcb_clear_area(conn, 0, win, 0, 0, 0, 0);
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
