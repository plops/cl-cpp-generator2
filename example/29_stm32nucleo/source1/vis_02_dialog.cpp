
#include "utils.h"

#include "globals.h"

extern State state;
#include "vis_01_serial.hpp"
#include "vis_02_dialog.hpp"
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSerialPortInfo>
#include <QSpinBox>
Dialog::Dialog(QWidget *parent)
    : QDialog(parent), m_serialPortLabel(new QLabel(tr("Serial port:"))),
      m_serialPortComboBox(new QComboBox),
      m_waitRequestLabel(new QLabel(tr("Wait request, msec:"))),
      m_waitRequestSpinBox(new QSpinBox),
      m_responseLabel(new QLabel(tr("Response:"))),
      m_responseLineEdit(new QLineEdit(tr("hello ... "))),
      m_trafficLabel(new QLabel(tr("No traffic."))),
      m_statusLabel(new QLabel(tr("Status: Not running."))),
      m_runButton(new QPushButton(tr("Start"))) {}
void Dialog::startReader() {}
void Dialog::showRequest(QString &s) {}
void Dialog::processError(QString &s) {}
void Dialog::processTimeout(QString &s) {}
void Dialog::activateRunButton(QString &s) {}