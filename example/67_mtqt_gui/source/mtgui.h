#include <QApplication>
#include <QEvent>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
;
class AnyQAppLambda {
public:
  virtual void run() = 0;
  virtual ~AnyQAppLambda();
};
class AnyQAppLambdaEvent : public QEvent {
public:
  AnyQAppLambda *al = nullptr;
  AnyQAppLambdaEvent(AnyQAppLambda *al);
  virtual ~AnyQAppLambdaEvent();
};
class BlockingEvent : public AnyQAppLambdaEvent {
public:
  std::atomic<bool> *done;
  BlockingEvent(AnyQAppLambda *al, std::atomic<bool> *done);
  ~BlockingEvent();
};
class QApplicationManager {
public:
  std::shared_ptr<std::atomic<bool>> done =
      std::make_shared<std::atomic<bool>>(false);
  bool we_own_app = true;
  std::thread thr;
  QCoreApplication *app = nullptr;
  ~QApplicationManager();
  static std::shared_ptr<QApplicationManager> create(int argc, char **argv);
  void wait_for_finished();
};

std::shared_ptr<QApplicationManager>
qapplication_manager(int argc = 0, char **argv = nullptr);

QCoreApplication *qapplication(int argc = 0, char **argv = nullptr);

void wait_for_qapp_to_finish();

void run_in_gui_thread(AnyQAppLambda *re);

void run_in_gui_thread_blocking(AnyQAppLambda *re);

void quit();

unsigned char wait_key();
