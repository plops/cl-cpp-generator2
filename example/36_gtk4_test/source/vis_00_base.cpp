
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_00_base.hpp"

Example_ListView_AppLauncher::Example_ListView_AppLauncher() {
  set_default_size(640, 320);
  set_title("app-launcher");
  auto factory = Gtk::SignalListItemFactory::create();
  factory->signal_setup().connect(
      sigc::mem_fun(*this, &Example_ListView_AppLauncher::setup_listitem));
  factory->signal_bind().connect(
      sigc::mem_fun(*this, &Example_ListView_AppLauncher::bind_listitem));
  auto model = create_application_list();
  m_list = Gtk::make_managed<Gtk::ListView>(Gtk::SingleSelection::create(model),
                                            factory);
  m_list->signal_activate().connect(
      sigc::mem_fun(*this, &Example_ListView_AppLauncher::activate));
  auto sw = Gtk::make_managed<Gtk::ScrolledWindow>();
  set_child(*sw);
  sw->set_child(*m_list);
}
Example_ListView_AppLauncher::~Example_ListView_AppLauncher() {}
Glib::RefPtr<Gio::ListModel>
Example_ListView_AppLauncher::create_application_list() {
  auto store = Gio::ListStore<Gio::AppInfo>::create();
  for (auto app : Gio::AppInfo::get_all()) {
    store->append(app);
  }
  return store;
}
void Example_ListView_AppLauncher::setup_listitem(
    const Glib::RefPtr<Gtk::ListItem> &item) {
  auto box = Gtk::make_managed<Gtk::Box>(Gtk::Orientation::HORIZONTAL, 12);
  auto image = Gtk::make_managed<Gtk::Image>();
  auto label = Gtk::make_managed<Gtk::Label>();
  image->set_icon_size(Gtk::IconSize::LARGE);
  box->append(*image);
  box->append(*label);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("setup") << (" ")
      << (std::setw(8)) << (" box='") << (box) << ("'") << (std::endl)
      << (std::flush);
  item->set_child(*box);
}
void Example_ListView_AppLauncher::bind_listitem(
    const Glib::RefPtr<Gtk::ListItem> &item) {
  auto image = dynamic_cast<Gtk::Image *>(item->get_child()->get_first_child());
  if (image) {
    auto label = dynamic_cast<Gtk::Label *>(image->get_next_sibling());
    if (label) {
      auto app_info = std::dynamic_pointer_cast<Gio::AppInfo>(item->get_item());
      if (app_info) {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("bind") << (" ") << (std::setw(8))
                    << (" app_info->get_display_name()='")
                    << (app_info->get_display_name()) << ("'") << (std::endl)
                    << (std::flush);
        image->set(app_info->get_icon());
        label->set_label(app_info->get_display_name());
      }
    }
  }
}
void Example_ListView_AppLauncher::activate(guint position) {
  auto item = std::dynamic_pointer_cast<Gio::ListModel>(m_list->get_model())
                  ->get_object(position);
  auto app_info = std::dynamic_pointer_cast<Gio::AppInfo>(item);
  if (app_info) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("launch") << (" ") << (std::setw(8))
                << (" app_info->get_display_name()='")
                << (app_info->get_display_name()) << ("'") << (std::endl)
                << (std::flush);
  }
}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto app = Gtk::Application::create();
  Example_ListView_AppLauncher hw;
  app->run(hw);
}