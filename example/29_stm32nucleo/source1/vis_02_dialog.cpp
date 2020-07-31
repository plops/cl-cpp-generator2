
#include "utils.h"

#include "globals.h"

extern State state;
#include "vis_01_serial.hpp"
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSerialPortInfo>
#include <QSpinBox>
Dialog::Dialog(QObject *parent) : QThread(parent) {}
void Dialog::startReader(){};