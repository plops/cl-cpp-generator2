
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

PenroseExtraInit::PenroseExtraInit(const Glib::ustring &css_name)
    : Glib::ExtraClassInit(
          [](void *g_class, void *class_data) {
            g_return_if_fail(GTK_IS_WIDGET_CLASS(g_class));
            auto klass = static_cast<GtkWidgetClass *>(g_class);
            auto css_name2 = static_cast<Glib::ustring *>(class_data);
            gtk_widget_class_set_css_name(klass, css_name2->c_str());
          },
          &m_css_name,
          [](GTypeInstance *instance, void *g_class) {
            g_return_if_fail(GTK_IS_WIDGET(instance));
          }) {}
PenroseWidget::PenroseWidget()
    : Glib::ObjectBase("PenroseWidget"),
      PenroseExtraInit("penrose-widget"), Gtk::Widget(), m_padding() {
  set_hexpand(true);
  set_vexpand(true);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("gtype name") << (" ")
      << (std::setw(8)) << (" G_OBJECT_TYPE_NAME(gobj())='")
      << (G_OBJECT_TYPE_NAME(gobj())) << ("'") << (std::endl) << (std::flush);
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
void PenroseWidget::measure_vfunc(Gtk::Orientation orientation, int for_size,
                                  int &minimum, int &natural,
                                  int &minimum_baseline,
                                  int &natural_baseline) {
  if ((Gtk::Orientation::HORIZONTAL) == (orientation)) {
    minimum = 60;
    natural = 100;
  } else {
    minimum = 50;
    natural = 70;
  }
  minimum_baseline = -1;
  natural_baseline = -1;
}
void PenroseWidget::on_map() { Gtk::Widget::on_map(); }
void PenroseWidget::on_unmap() { Gtk::Widget::on_unmap(); }
void PenroseWidget::on_realize() { Gtk::Widget::on_realize(); }
void PenroseWidget::on_unrealize() { Gtk::Widget::on_unrealize(); }
void PenroseWidget::snapshot_vfunc(
    const Glib::RefPtr<Gtk::Snapshot> &snapshot) {
  auto allocation = get_allocation();
  auto rect =
      Gdk::Rectangle(0, 0, allocation.get_width(), allocation.get_height());
  auto style = get_style_context();
  auto cr = snapshot->append_cairo(rect);
  style->render_background(cr, 0, 0, allocation.get_width(),
                           allocation.get_height());
  cr->move_to(0, 0);
  cr->line_to(0, 100);
  cr->stroke();
}
void PenroseWidget::on_parsing_error(
    const Glib::RefPtr<Gtk::CssSection> &section, const Glib::Error &error) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("parse") << (" ")
      << (std::setw(8)) << (" error.what()='") << (error.what()) << ("'")
      << (std::setw(8)) << (" section->get_file()->get_uri()='")
      << (section->get_file()->get_uri()) << ("'") << (std::setw(8))
      << (" section->get_start_location()='") << (section->get_start_location())
      << ("'") << (std::setw(8)) << (" section->get_end_location()='")
      << (section->get_end_location()) << ("'") << (std::endl) << (std::flush);
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
void ExampleWindow::on_button_quit() { hide(); }
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