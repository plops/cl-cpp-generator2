
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>
// implementation
#include "vis_02_drawing_widget.hpp"

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
                                              int &natural_width) {
  minimum_width = 60;
  natural_width = 100;
}
void PenroseWidget::get_preferred_height_for_width_vfunc(int width,
                                                         int &minimum_height,
                                                         int &natural_height) {
  minimum_height = 50;
  natural_height = 70;
}
void PenroseWidget::get_preferred_height_vfunc(int &minimum_height,
                                               int &natural_height) {
  minimum_height = 50;
  natural_height = 70;
}
void PenroseWidget::get_preferred_width_for_height_vfunc(int height,
                                                         int &minimum_height,
                                                         int &natural_height) {
  minimum_height = 60;
  natural_height = 100;
}
void PenroseWidget::on_size_allocate(Gtk::Allocation &allocation) {
  set_allocation(allocation);
  if (m_refGdkWindow) {
    m_refGdkWindow->move_resize(allocation.get_x(), allocation.get_y(),
                                allocation.get_width(),
                                allocation.get_height());
  }
}
void PenroseWidget::on_map() { Gtk::Widget::on_map(); }
void PenroseWidget::on_unmap() { Gtk::Widget::on_unmap(); }
void PenroseWidget::on_realize() {
  set_realized();
  m_scale = m_scale_prop.get_value();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" m_scale='") << (m_scale) << ("'") << (std::endl)
      << (std::flush);
  if (!(m_refGdkWindow)) {
    GdkWindowAttr attr;
    memset(&attr, 0, sizeof(attr));
    auto allocation = get_allocation();
    attr.x = allocation.get_x();
    attr.y = allocation.get_y();
    attr.width = allocation.get_width();
    attr.height = allocation.get_height();
    attr.event_mask = ((get_events()) | (Gdk::EXPOSURE_MASK));
    attr.window_type = GDK_WINDOW_CHILD;
    attr.wclass = GDK_INPUT_OUTPUT;
    m_refGdkWindow = Gdk::Window::create(get_parent_window(), &attr,
                                         ((GDK_WA_X) | (GDK_WA_Y)));
    set_window(m_refGdkWindow);
    m_refGdkWindow->set_user_data(gobj());
  }
}
void PenroseWidget::on_unrealize() {
  if (m_refGdkWindow) {
    m_refGdkWindow.reset();
  }
  Gtk::Widget::on_unrealize();
}
bool PenroseWidget::on_draw(const Cairo::RefPtr<Cairo::Context> &cr) {
  auto allocation = get_allocation();
  auto scale_x = ((static_cast<double>(allocation.get_width())) / (m_scale));
  auto scale_y = ((static_cast<double>(allocation.get_height())) / (m_scale));
  auto style = get_style_context();
  style->render_background(cr, allocation.get_x(), allocation.get_y(),
                           allocation.get_width(), allocation.get_height());
  auto state2 = style->get_state();
  Gdk::Cairo::set_source_rgba(cr, style->get_color(state2));
  cr->move_to((((155.)) * (scale_x)), (((165.)) * (scale_y)));
  cr->line_to((((155.)) * (scale_x)), (((838.)) * (scale_y)));
  cr->line_to((((265.)) * (scale_x)), (((9.00e+2)) * (scale_y)));
  cr->line_to((((849.)) * (scale_x)), (((564.)) * (scale_y)));
  cr->line_to((((849.)) * (scale_x)), (((438.)) * (scale_y)));
  cr->line_to((((265.)) * (scale_x)), (((1.00e+2)) * (scale_y)));
  cr->line_to((((155.)) * (scale_x)), (((165.)) * (scale_y)));
  cr->move_to((((265.)) * (scale_x)), (((1.00e+2)) * (scale_y)));
  cr->line_to((((265.)) * (scale_x)), (((652.)) * (scale_y)));
  cr->line_to((((526.)) * (scale_x)), (((502.)) * (scale_y)));
  cr->move_to((((369.)) * (scale_x)), (((411.)) * (scale_y)));
  cr->line_to((((633.)) * (scale_x)), (((564.)) * (scale_y)));
  cr->move_to((((369.)) * (scale_x)), (((286.)) * (scale_y)));
  cr->line_to((((369.)) * (scale_x)), (((592.)) * (scale_y)));
  cr->move_to((((369.)) * (scale_x)), (((286.)) * (scale_y)));
  cr->line_to((((849.)) * (scale_x)), (((564.)) * (scale_y)));
  cr->move_to((((633.)) * (scale_x)), (((564.)) * (scale_y)));
  cr->line_to((((155.)) * (scale_x)), (((838.)) * (scale_y)));
  cr->stroke();
  return true;
}
void PenroseWidget::on_parsing_error(
    const Glib::RefPtr<const Gtk::CssSection> &section,
    const Glib::Error &error) {

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