
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
  auto uri = "https://www.youtube.com";
  if ((2) == (argc)) {
    uri = argv[1];
    argc = 1;
  }
  Glib::set_application_name("gtkmm-plplot-test13");
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm-plplot.example");
  Gtk::Window win;
  auto webview = new Window;
  auto setting = webkit_web_view_get_settings(WEBKIT_WEB_VIEW(webview));
  g_object_set(G_OBJECT(setting), "enable-developer-extras", true, nullptr);
  auto inspector = webkit_web_view_get_inspector(WEBKIT_WEB_VIEW(webview));
  webkit_web_inspector_show(WEBKIT_WEB_INSPECTOR(inspector));
  win.add(*webview);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" uri='") << (uri) << ("'") << (std::endl)
      << (std::flush);
  webview->load_uri(uri);
  win.show_all();
  app->run(win);
}