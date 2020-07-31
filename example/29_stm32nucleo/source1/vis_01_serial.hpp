#ifndef VIS_01_SERIAL_H
#define VIS_01_SERIAL_H
#include "utils.h"
;
#include "globals.h"
;
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtSerialPort/QSerialPort>
#include <QTime>
;
#include <QMutex>
#include <QThread>
#include <QWaitCondition>
;
#include <QMutex>
#include <QThread>
#include <QWaitCondition>;
class SerialReaderThread : public QThread {
        Q_OBJECT
        public:
        explicit  SerialReaderThread (QObject* parent = nullptr)  ;  
         ~SerialReaderThread ()  ;  
        void startReader (const QString& portName, int waitTimeout, const QString& response)  ;  
            signals:
        void error (const QString& s)  ;  
    void timeout (const QString& s)  ;  
        private:
        void run ()  ;  
        QString m_portName;
        QString m_response;
        int m_waitTimeout = 0;
        QMutex m_mutex;
        bool m_quit = false;
};
#endif