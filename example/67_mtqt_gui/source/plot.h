#include <jkqtplotter/graphs/jkqtpscatter.h>
#include <jkqtplotter/jkqtplotter.h>
#include <mtgui.h>
#include <mtgui_template.h>
;
#include <memory>
#include <mutex>
#include <thread>
;
class Figure {
public:
  JKQTPlotter plot;
  std::map<std::string, JKQTPXYLineGraph *> graphs;
  JKQTPXYLineGraph *labeled_graph(std::string label);
  void set(const std::vector<double> &xs, const std::vector<double> &ys,
           std::string label);
};
class Plotter {
public:
  void clear_plot(std::string title);
  void delete_plot(std::string title);
  void plot(const std::vector<double> &xs, const std::vector<double> &ys,
            std::string title, std::string label);
  static std::shared_ptr<Plotter> create();
  Plotter();
  std::weak_ptr<Plotter> self;
  std::mutex call_plot_mutex;
  void call_plot(const std::vector<double> &xs, const std::vector<double> &ys,
                 std::string title, std::string label);
  std::map<std::string, std::shared_ptr<Figure>> plots_;
  std::shared_ptr<Figure> named_plot(std::string name);
  void clear_plot_internal(std::string title);
  void delete_plot_internal(std::string title);
  std::mutex plot_internal_mtx;
  void plot_internal(const std::vector<double> &xs,
                     const std::vector<double> &ys, std::string title,
                     std::string label);
};

std::shared_ptr<Plotter> plotter();

void clear_plot(std::string title);

void delete_plot(std::string title);

void plot(const std::vector<double> &ys, std::string title, std::string label);

void plot(const std::vector<double> &xs, const std::vector<double> &ys,
          std::string title, std::string label);

void initialize_plotter();
