
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

PenroseWidget::PenroseWidget()
    : Glib::ObjectBase("PenroseWidget"), Gtk::Widget(),
      m_scale_prop(*this, "example_scale", 500), m_scale(1000) {
  set_has_window(true);
  set_name("penrose-widget");
  m_refCssProvider = Gtk::CssProvider::create();
  auto style = get_style_context();
  style->add_provider(m_refCssProvider,
                      GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
  m_refCssProvider->signal_parsing_error().connect(
      sigc::mem_fun(*this, &PenroseWidget::on_parsing_error));
  m_refCssProvider->load_from_path("custom_gtk.css");
}
PenroseWidget::~PenroseWidget() {}
Gtk::SizeRequestMode PenroseWidget::get_request_mode_vfunc() {
  return Gtk::Widget::get_request_mode_vfunc();
}
void PenroseWidget::get_preferred_width_vfunc(int &minimum_width,
                                              int &natural_width) {}
void PenroseWidget::get_preferred_height_for_width_vfunc(int width,
                                                         int &minimum_height,
                                                         int &natural_height) {}
void PenroseWidget::get_preferred_height_vfunc(int &minimum_height,
                                               int &natural_height) {}
void PenroseWidget::get_preferred_width_for_height_vfunc(int height,
                                                         int &minimum_height,
                                                         int &natural_height) {}
void PenroseWidget::on_size_allocate(Gtk::Allocation &allocation) {}
void PenroseWidget::on_map() { Gtk::Widget::on_map(); }
void PenroseWidget::on_unmap() { Gtk::Widget::on_unmap(); }
void PenroseWidget::on_realize() { Gtk::Widget::on_realize(); }
void PenroseWidget::on_unrealize() { Gtk::Widget::on_unrealize(); }
bool PenroseWidget::on_draw(const Cairo::RefPtr<Cairo::Context> &cr) {}
void PenroseWidget::on_parsing_error(
    const Glib::RefPtr<Gtk::CssSection> &section, const Glib::Error &error) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("parse") << (" ")
      << (std::setw(8)) << (" error.what()='") << (error.what()) << ("'")
      << (std::setw(8)) << (" section->get_file()->get_uri()='")
      << (section->get_file()->get_uri()) << ("'") << (std::endl)
      << (std::flush);
}
ExampleWindow::ExampleWindow() {
  set_title("custom widget example");
  set_default_size(600, 400);
  m_grid.set_margin(6);
  m_grid.set_row_spacing(10);
  m_grid.set_column_spacing(10);
  add(m_grid);
  m_grid.attach(m_penrose, 0, 0);
}
ExampleWindow::~ExampleWindow() {}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto app = Gtk::Application::create();
  ExampleWindow hw;
  app->run(hw);
}