
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
void Window::load_uri(const gchar *uri) {
  webkit_web_view_load_uri(*this, uri);
}
int main(int argc, char **argv) {
  Glib::set_application_name("gtkmm-plplot-test13");
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm-plplot.example");
  Window win;
  win.load_uri("https://www.google.com");
  app->run(win);
}