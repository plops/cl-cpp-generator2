
#include "utils.h"

#include "globals.h"

;
// https://doc.qt.io/qt-5/qtserialport-blockingmaster-example.html
;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>

using namespace std::chrono_literals;
State state = {};
SerialReaderThread::SerialReaderThread(QObject *parent) : QThread(parent) {}
SerialReaderThread::~SerialReaderThread() {
  m_mutex.lock();
  m_quit = true;
  m_mutex.unlock();
  wait();
}
void SerialReaderThread::startReader(const QString &portName, int waitTimeout,
                                     const QString &response) {
  const QMutexLocker locker(&m_mutex);
  m_portName = portName;
  m_waitTimeout = waitTimeout;
  m_response = response;
  if (!(isRunning())) {
    start();
  }
}
void SerialReaderThread::request(const QString &s) {}
void SerialReaderThread::error(const QString &s) {}
void SerialReaderThread::timeout(const QString &s) {}
void SerialReaderThread::run() {
  bool currentPortNameChanged = false;
  m_mutex.lock();
  QString currentPortName;
  if (!((currentPortName) == (m_portName))) {
    currentPortName = m_portName;
    currentPortNameChanged = true;
  }
  auto currentWaitTimeout = m_waitTimeout;
  auto currentResponse = m_response;
};
int main(int argc, char **argv) {
  state._main_version = "54a25292f143d0cdee976744c74ea5ca87f43c54";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/27_sparse_eigen_hydrogen";
  state._code_generation_time = "23:22:15 of Thursday, 2020-07-30 (GMT+1)";
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::setw(8)) << (" state._main_version='") << (state._main_version)
      << ("'") << (std::endl) << (std::flush);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" state._code_repository='")
      << (state._code_repository) << ("'") << (std::endl) << (std::flush);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
      << (std::setw(8)) << (" state._code_generation_time='")
      << (state._code_generation_time) << ("'") << (std::endl) << (std::flush);
  auto app(argc, argv);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};