#include <chrono>
#include <iomanip>
#include <iostream>
;
#include <QApplication>
#include <QMainWindow>
#include <qcustomplot.h>
;
class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  QCustomPlot *plot_;
  int graph_count_;
  explicit MainWindow(QWidget *parent = 0);
  void plot_scatter(std::vector<double> x, std::vector<double> y);
  void plot_line(std::vector<double> x, std::vector<double> y);
  ~MainWindow();
};
