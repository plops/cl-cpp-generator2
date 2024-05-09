#include "moc_gui.h"
MainWindow::MainWindow(QWidget *parent) : graph_count_{0}, QMainWindow{parent} {
  (plot_) = (new QCustomPlot(this));
  setCentralWidget(plot_);
  setGeometry(400, 250, 542, 390);
}
void MainWindow::plot_scatter(std::vector<double> x, std::vector<double> y) {
  assert((x.size()) == (y.size()));
  QVector<double> qx(x.size()), qy(y.size());
  for (auto i = 0; (i) < (x.size()); (i) += (1)) {
    ((qx)[(i)]) = ((x)[(i)]);
  }
  for (auto i = 0; (i) < (y.size()); (i) += (1)) {
    ((qy)[(i)]) = ((y)[(i)]);
  }
  plot_->addGraph();
  (plot_)->(graph(graph_count_))->(setData(qx, qy));
  (plot_)->(graph(graph_count_))->(setLineStyle(QCPGraph::lsNone));
  (plot_)
      ->(graph(graph_count_))
      ->(setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 4)));
  (graph_count_)++;
}
void MainWindow::plot_line(std::vector<double> x, std::vector<double> y) {
  assert((x.size()) == (y.size()));
  QVector<double> qx(x.size()), qy(y.size());
  for (auto i = 0; (i) < (x.size()); (i) += (1)) {
    ((qx)[(i)]) = ((x)[(i)]);
  }
  for (auto i = 0; (i) < (y.size()); (i) += (1)) {
    ((qy)[(i)]) = ((y)[(i)]);
  }
  plot_->addGraph();
  (plot_)->(graph(graph_count_))->(setData(qx, qy));
  (graph_count_)++;
}
MainWindow::~MainWindow() {}