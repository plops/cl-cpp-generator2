
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
Dialog::Dialog(QObject *parent) : QThread(parent) {}
void Dialog::startReader() {}
void Dialog::showRequest(QString &s) {}
void Dialog::processError(QString &s) {}
void Dialog::processTimeout(QString &s) {}
void Dialog::activateRunButton(QString &s){};