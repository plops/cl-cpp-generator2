
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

Window::Window() : canvas() {
  set_default_size(720, 580);
  Gdk::Geometry geom;
  auto aspect = (((7.20e+2)) / ((5.80e+2)));
  geom.min_aspect = aspect;
  geom.max_aspect = aspect;
  set_geometry_hints(*this, geom, Gdk::HINT_ASPECT);
  set_title("plplot test");
  canvas.set_hexpand(true);
  canvas.set_vexpand(true);
  add_plot_1();
  grid.attach(canvas, 0, 0, 1, 1);
  grid.set_row_spacing(5);
  grid.set_column_spacing(5);
  grid.set_column_homogeneous(false);
  add(grid);
  set_border_width(10);
  grid.show_all();
}
Window::~Window() {}
void Window::add_plot_1() {
  if (!((nullptr) == (plot))) {
    canvas.remove_plot(*plot);
    plot = nullptr;
  }
  auto npts = 73;
  auto x_va = std::valarray<double>(npts);
  auto y_va = std::valarray<double>(npts);
  auto xmin = 0;
  auto xmax = ((60) * (60) * (24));
  auto ymin = 10;
  auto ymax = 20;
  for (auto i = 0; (i) < (npts); (i) += (1)) {
    x_va[i] =
        ((static_cast<double>(((xmax) * (i)))) / (static_cast<double>(npts)));
    y_va[i] = (((15.)) -
               ((((5.0)) * (cos((((2.0)) * ((3.14159265358979300000000000000)) *
                                 (((static_cast<double>(i)) /
                                   (static_cast<double>(npts))))))))));
  }
  auto plot_data = Gtk::manage(new Gtk::PLplot::PlotData2D(
      x_va, y_va, Gdk::RGBA("blue"), Gtk::PLplot::LineStyle::LONG_DASH_LONG_GAP,
      (5.0)));
  plot = Gtk::manage(new Gtk::PLplot::Plot2D(*plot_data));
  plot->set_axis_time_format_x("%H:%M");
  canvas.add_plot(*plot);
  plot->hide_legend();
}
int main(int argc, char **argv) {
  Glib::set_application_name("gtkmm-plplot-test13");
  auto app = Gtk::Application::create(argc, argv, "org.gtkmm-plplot.example");
  Window win;
  app->run(win);
}