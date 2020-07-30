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
        signals:
        void request (const QString& s)  ;  ;
        void error (const QString& s)  ;  ;
        void timeout (const QString& s)  ;  ;
        private:
        void run ()  ;  ;
        QString m_portName;
        QString m_response;
        int m_waitTimeout = 0;
        QMutex m_mutex;
        bool m_quit = false;
};
int main (int argc, char** argv)  ;  
#endif