#include <QApplication>
#include <QMainWindow>
#include <chrono>
#include <cmath>
#include <plot.h>
#include <thread>

void thread_independent_qt_gui_app() {
  // no need to initialize qt

  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("first window") << (" ") << (std::endl) << (std::flush);
  }
}

int main(int argc, char **argv) {
  auto xs{std::vector<double>(100)};
  auto ys{std::vector<double>(100)};
  for (auto q = 0; (q) < (100); (q) += (1)) {
    for (auto i = 0; (i) < (100); (i) += (1)) {
      ((xs)[(i)]) = (i);
      ((ys)[(i)]) =
          ((exp((-1.00e-2F) * (i))) * (sin((0.40F) * ((i) + ((5.0F) * (q))))));
    }
    plot(xs, ys, "bla1", "bla2");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }
  wait_for_qapp_to_finish();
  return 0;
}
