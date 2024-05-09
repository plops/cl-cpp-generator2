#include <memory>
#include <mutex>
#include <plot.h>
#include <thread>
JKQTPXYLineGraph *Figure::labeled_graph(std::string label) {
  auto it{graphs.find(label)};
  if (!((graphs.end()) == (it))) {
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("found") << (" ") << (std::setw(8))
                  << (" label='") << (label) << ("'") << (std::endl)
                  << (std::flush);
    }
    return it->second;
  }
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("create") << (" ") << (std::setw(8)) << (" label='")
                << (label) << ("'") << (std::endl) << (std::flush);
  }
  auto graph{new JKQTPXYLineGraph(&plot)};
  graph->setTitle(QObject::tr(label.c_str()));
  ((graphs)[(label)]) = (graph);
  return graph;
}
void Figure::set(const std::vector<double> &xs, const std::vector<double> &ys,
                 std::string label) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto ds{plot.getDatastore()};
  auto X{QVector<double>(xs.size())};
  auto Y{QVector<double>(ys.size())};
  auto contains_nan{false};
  for (auto i = 0; (i) < (std::min(xs.size(), ys.size())); (i) += (1)) {
    if (std::isnan(((xs)[(i)]) + ((ys)[(i)]))) {
      (contains_nan) = (true);
      continue;
    }
    (X) << ((xs)[(i)]);
    (Y) << ((ys)[(i)]);
  }
  auto columnX{ds->addCopiedColumn(X, "x")};
  auto columnY{ds->addCopiedColumn(Y, "y")};
  auto graph{labeled_graph(label)};
  graph->setXColumn(columnX);
  graph->setYColumn(columnY);
  plot.addGraph(graph);
  plot.zoomToFit();
}
void Plotter::clear_plot(std::string title) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto plotter{self.lock()};
  run_in_gui_thread(new QAppLambda(
      [plotter, title]() { plotter->clear_plot_internal(title); }));
}
void Plotter::delete_plot(std::string title) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto plotter{self.lock()};
  run_in_gui_thread(new QAppLambda(
      [plotter, title]() { plotter->delete_plot_internal(title); }));
}
void Plotter::plot(const std::vector<double> &xs, const std::vector<double> &ys,
                   std::string title, std::string label) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  call_plot(xs, ys, title, label);
}
std::shared_ptr<Plotter> Plotter::create() {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto plotter{std::shared_ptr<Plotter>(new Plotter)};
  (plotter->self) = (plotter);
  return plotter;
}
Plotter::Plotter() {}
void Plotter::call_plot(const std::vector<double> &xs,
                        const std::vector<double> &ys, std::string title,
                        std::string label) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto ul{std::unique_lock<std::mutex>(call_plot_mutex)};
  auto plotter{self.lock()};
  run_in_gui_thread(new QAppLambda([plotter, xs, ys, title, label]() {
    plotter->plot_internal(xs, ys, title, label);
  }));
}
std::shared_ptr<Figure> Plotter::named_plot(std::string name) {
  auto it{plots_.find(name)};
  if (!((it) == (plots_.end()))) {
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("found existing") << (" ") << (std::setw(8))
                  << (" name='") << (name) << ("'") << (std::endl)
                  << (std::flush);
    }
    return it->second;
  }
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("create new figure") << (" ") << (std::setw(8))
                << (" name='") << (name) << ("'") << (std::endl)
                << (std::flush);
  }
  auto figure{std::make_shared<Figure>()};
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("store figure") << (" ") << (std::endl) << (std::flush);
  }
  ((plots_)[(name)]) = (figure);
  auto &plot{figure->plot};
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("set title..") << (" ") << (std::endl) << (std::flush);
  }
  plot.setWindowTitle(QString(name.c_str()));
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("show..") << (" ") << (std::endl) << (std::flush);
  }
  plot.show();
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("resize..") << (" ") << (std::endl) << (std::flush);
  }
  plot.resize(600, 400);
  return figure;
}
void Plotter::clear_plot_internal(std::string title) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto it{plots_.find(title)};
  if (!((it) == (plots_.end()))) {
    (it)->(second)->(plot.clearGraphs());
    auto plot_use_count{(it)->(second).use_count()};
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("\033[1;31m CLEAR \033[0m") << (" ")
                  << (std::setw(8)) << (" (it)->(first)='") << ((it)->(first))
                  << ("'") << (std::setw(8)) << (" plot_use_count='")
                  << (plot_use_count) << ("'") << (std::endl) << (std::flush);
    }
  }
}
void Plotter::delete_plot_internal(std::string title) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto it{plots_.find(title)};
  if (!((it) == (plots_.end()))) {
    ((it)->(second)) = (nullptr);
    auto plot_use_count{(it)->(second).use_count()};
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("\033[1;31m CLEAR \033[0m") << (" ")
                  << (std::setw(8)) << (" (it)->(first)='") << ((it)->(first))
                  << ("'") << (std::setw(8)) << (" plot_use_count='")
                  << (plot_use_count) << ("'") << (std::endl) << (std::flush);
    }
  }
}
void Plotter::plot_internal(const std::vector<double> &xs,
                            const std::vector<double> &ys, std::string title,
                            std::string label) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto ul{std::unique_lock<std::mutex>(plot_internal_mtx)};
  auto figure{std::shared_ptr<Figure>(named_plot(title))};
  figure->set(xs, ys, label);
}
std::mutex plotter_mtx;
std::shared_ptr<Plotter> plotter_ = nullptr;

std::shared_ptr<Plotter> plotter() {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto ul{std::unique_lock<std::mutex>(plotter_mtx)};
  if ((nullptr) == (plotter_)) {
    (plotter_) = (Plotter::create());
  }
  return plotter_;
}

void clear_plot(std::string title) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  (plotter())->(clear_plot(title));
}

void delete_plot(std::string title) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  (plotter())->(delete_plot(title));
}

void plot(const std::vector<double> &ys, std::string title, std::string label) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto indexes{std::vector<double>(ys.size())};
  for (auto i = 0; (i) < (ys.size()); (i) += (1)) {
    indexes.push_back(i);
  }
  (plotter())->(plot(indexes, ys, title, label));
}

void plot(const std::vector<double> &xs, const std::vector<double> &ys,
          std::string title, std::string label) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  (plotter())->(plot(xs, ys, title, label));
}

void initialize_plotter() {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  plotter();
}
