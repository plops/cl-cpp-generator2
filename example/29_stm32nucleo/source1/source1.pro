######################################################################
# Automatically generated by qmake (3.1) Sun Aug 2 14:42:59 2020
######################################################################
QT += serialport widgets

TEMPLATE = app
TARGET = source1
INCLUDEPATH += .

# You can make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# Please consult the documentation of the deprecated API in order to know
# how to port your code away from it.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# Input
HEADERS += globals.h \
           pb.h \
           pb_common.h \
           pb_decode.h \
           pb_encode.h \
           simple.pb.h \
           ui_nucleo_l476rg_adc.h \
           utils.h \
           vis_00_main.hpp \
           vis_01_serial.hpp \
           vis_02_dialog.hpp
FORMS += nucleo_l476rg_adc.ui
SOURCES += pb_common.c \
           pb_decode.c \
           pb_encode.c \
           simple.pb.c \
           vis_00_main.cpp \
           vis_01_serial.cpp \
           vis_02_dialog.cpp
