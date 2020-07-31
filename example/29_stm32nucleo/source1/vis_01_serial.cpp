
#include "utils.h"

#include "globals.h"

extern State state;
// https://doc.qt.io/qt-5/qtserialport-blockingmaster-example.html
;
#include "vis_01_serial.hpp"
#include <QTime>
#include <QtSerialPort/QSerialPort>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
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
}