
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

Window::Window() : Gtk::Widget(webkit_web_view_new()) {}
Window::~Window() {}
Window::operator WebKitWebView *() { return WEBKIT_WEB_VIEW(gobj()); }
void Window::load_uri(const gchar *uri) {
  webkit_web_view_load_uri(*this, uri);
}
int main(int argc, char **argv) {
  Glib::set_application_name("gtkmm-plplot-test13");
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm-plplot.example");
  Gtk::Window win;
  auto webview = new Window;
  win.add(*webview);
  webview->load_uri("https://www.youtube.com");
  win.show_all();
  app->run(win);
}