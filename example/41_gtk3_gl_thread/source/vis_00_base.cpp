
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

GraphicsArea::GraphicsArea() {
  set_title("graphics-area");
  set_default_size(640, 360);
  add(vbox);
  area.set_hexpand(true);
  area.set_vexpand(true);
  area.set_auto_render(true);
  vbox.add(area);
  area.signal_render().connect(sigc::mem_fun(*this, &GraphicsArea::render));
  area.show();
  vbox.show();
}
GraphicsArea::~GraphicsArea() {}
void GraphicsArea::run() {
  while (true) {
    dispatcher.emit();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}
void GraphicsArea::onNotifcationFromThread() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::endl) << (std::flush);
  queue_draw();
}
bool GraphicsArea::render(const Glib::RefPtr<Gdk::GLContext> &ctx) {
  area.throw_if_error();
  glClearColor((0.50f), (0.50f), (0.50f), (1.0f));
  glClear(GL_COLOR_BUFFER_BIT);
  glFlush();
  return true;
}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto app = Gtk::Application::create(argc, argv, "gtk3-gl-threads");
  GraphicsArea hw;
  std::thread th(&GraphicsArea::run, &hw);
  app->run(hw);
}