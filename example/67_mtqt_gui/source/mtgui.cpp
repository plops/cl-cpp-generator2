#include <mtgui.h>
#include <mtgui_template.h>
AnyQAppLambda::~AnyQAppLambda() {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
}
AnyQAppLambdaEvent::AnyQAppLambdaEvent(AnyQAppLambda *al)
    : QEvent{QEvent::Type(48301)}, al{al} {}
AnyQAppLambdaEvent::~AnyQAppLambdaEvent() {
  if (!((nullptr) == (al))) {
    (al)->(run());
  }
  delete (al);
}
BlockingEvent::BlockingEvent(AnyQAppLambda *al, std::atomic<bool> *done)
    : AnyQAppLambdaEvent{al}, done{done} {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
}
BlockingEvent::~BlockingEvent() {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  if (!((nullptr) == (this->al))) {
    (this)->(al)->(run());
  }
  delete (al);
  (al) = (nullptr);
  (done)->(store(true));
}
QApplicationManager::~QApplicationManager() {
  if (we_own_app) {
    quit();
    if (thr.joinable()) {
      thr.join();
    }
  }
}
std::shared_ptr<QApplicationManager> QApplicationManager::create(int argc,
                                                                 char **argv) {
  auto qm{std::make_shared<QApplicationManager>()};
  if (!((nullptr) == (QApplication::instance()))) {
    (qm->we_own_app) = (false);
    (qm->app) = (QCoreApplication::instance());
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("we are not managing this qapp instance.")
                  << (" ") << (std::endl) << (std::flush);
    }
    (qm)->(app)->(
        postEvent(qm->app, new AnyQAppLambdaEvent(new QAppLambda([qm]() {
                    QObject::connect(qm->app, &QApplication::aboutToQuit,
                                     qm->app, [qm]() { (*qm->done) = (true); });
                  }))));
    return qm;
  }
  std::atomic<bool> ready{false};
  (qm->thr) = (std::thread([&]() {
    (qm->app) = (new class QApplication(argc, argv));
    QObject::connect(qm->app, &QApplication::aboutToQuit, qm->app,
                     [qm]() { (*qm->done) = (true); });
    (ready) = (true);
    qm->app->exec();
  }));
  while (!(ready)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return qm;
}
void QApplicationManager::wait_for_finished() {
  while (!(*done)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}
std::mutex QApp_mtx;
std::shared_ptr<QApplicationManager> qm = nullptr;

std::shared_ptr<QApplicationManager> qapplication_manager(int argc,
                                                          char **argv) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("request lock") << (" ") << (std::endl) << (std::flush);
  }
  std::unique_lock<std::mutex> ul(QApp_mtx);
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("have lock") << (" ") << (std::endl) << (std::flush);
  }
  if ((nullptr) == (qm)) {
    (qm) = (QApplicationManager::create(argc, argv));
  }
  return qm;
}

QCoreApplication *qapplication(int argc, char **argv) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  return (qapplication_manager(argc, argv))->(app);
}

void wait_for_qapp_to_finish() {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  (qapplication_manager())->(wait_for_finished());
}

void run_in_gui_thread(AnyQAppLambda *re) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  auto qm{qapplication()};
  qm->postEvent(qm, new AnyQAppLambdaEvent(re));
}

void run_in_gui_thread_blocking(AnyQAppLambda *re) {
  {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::endl) << (std::flush);
  }
  std::atomic<bool> done{false};
  auto qm{qapplication()};
  qm->postEvent(qm, new BlockingEvent(re, &done));
  while (!(done)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void quit() {
  auto app{(qapplication_manager())->(app)};
  run_in_gui_thread_blocking(new QAppLambda([app]() { (app)->(quit()); }));
}

unsigned char wait_key() { return 0; }
