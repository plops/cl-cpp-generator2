g++ mtgui*.cpp \
    -I. \
    `pkg-config Qt5Widgets --cflags --libs`
