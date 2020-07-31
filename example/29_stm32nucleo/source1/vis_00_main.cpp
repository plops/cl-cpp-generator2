
#include "utils.h"

#include "globals.h"

// https://doc.qt.io/qt-5/qtserialport-blockingmaster-example.html
;
#include "vis_01_serial.hpp"
#include <QTime>
#include <QtSerialPort/QSerialPort>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
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
  m_mutex.unlock();
  QSerialPort serial;
  while (!(m_quit)) {
    if (currentPortNameChanged) {
      serial.close();
      serial.setPortName(currentPortName);
      if (!(serial.open(QIODevice::ReadWrite))) {
        emit error(tr("Cant open %1, error code %2")
                       .arg(m_portName)
                       .arg(serial.error()));
        return;
      }
      if (serial.waitForReadyRead(currentWaitTimeout)) {
        auto requestData = serial.readAll();
        while (serial.waitForReadyRead(10)) {
          (requestData) += (serial.readAll());
        }
      } else {
        emit timeout(tr("Wait read request timeout %1")
                         .arg(QTime::currentTime().toString()));
      }
      m_mutex.lock();
      if ((currentPortName) == (m_portName)) {
        currentPortNameChanged = false;
      } else {
        currentPortName = m_portName;
        currentPortNameChanged = true;
      }
      currentWaitTimeout = m_waitTimeout;
      currentResponse = m_response;
      m_mutex.unlock();
    }
  }
};
int main(int argc, char **argv) {
  state._main_version = "1b15346c0729cd40a7fda8db514926aca8881bfa";
  state._code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                           "master/example/27_sparse_eigen_hydrogen";
  state._code_generation_time = "06:11:32 of Friday, 2020-07-31 (GMT+1)";
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
  QApplication app(argc, argv);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
}