#ifndef VIS_02_DIALOG_H
#define VIS_02_DIALOG_H
#include "utils.h"
;
#include "globals.h"
;
#include <QtWidgets/QDialog>
;
#include <QtWidgets/QDialog>
QT_BEGIN_NAMESPACE;
class QLabel;
class QLineEdit;
class QComboBox;
class QSpinBox;
class QPushButton;
QT_END_NAMESPACE;;
class Dialog : public QDialog {
        Q_OBJECT;
        public:
        explicit  Dialog (QWidget* parent = nullptr)  ;  
        private slots:
        void startReader ()  ;  
        void showRequest (QString& s)  ;  
        void processError (QString& s)  ;  
        void processTimeout (QString& s)  ;  
        void activateRunButton (QString& s)  ;  
        private:
        int m_transactionCount=0;
        QLabel* m_serialPortLabel=nullptr;
        QComboBox* m_serialPortComboBox=nullptr;
        QLabel* m_waitRequestLabel=nullptr;
        QSpinBox* m_waitRequestSpinBox=nullptr;
        QLabel* m_responseLabel=nullptr;
        QLineEdit* m_responseLineEdit=nullptr;
        QLabel* m_trafficLabel=nullptr;
        QLabel* m_statusLabel=nullptr;
        QPushButton* m_runButton=nullptr;
};
#endif