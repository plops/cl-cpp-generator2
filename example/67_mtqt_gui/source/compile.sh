g++ mtgui.cpp \
    mtgui_driver.cpp \
    --std=c++17 \
    -I. \
    `pkg-config Qt5Widgets --cflags --libs`
