#ifndef VIS_00_MAIN_H
#define VIS_00_MAIN_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <thread>
;
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
;
class SerialReaderThread : public QThread {
        explicit  SerialReaderThread (QObject* parent)  ;  ;
         ~SerialReaderThread ()  ;  ;
        void startReader (const QString& portName, int waitTimeout, const QString& response)  ;  ;
};
int main (int argc, char** argv)  ;  
#endif