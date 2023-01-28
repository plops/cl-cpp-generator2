# C++20 modules with Qt

- try to speed up compilation with a module

- use the method from 107_clang_mod with the fatheader from 99_qt_pch

## install qt on fedora

```
dnf search qt5|grep devel|grep ^qt5 | tee f
cat f|grep ^qt|grep 64|cut -d ":" -f1|cut -d "." -f1|xargs 


sudo dnf install qt5-qt3d-devel qt5-qtbase-devel qt5-qtbase-private-devel qt5-qtcharts-devel qt5-qtconfiguration-devel qt5-qtconnectivity-devel qt5-qtdatavis3d-devel qt5-qtdeclarative-devel qt5-qtfeedback-devel qt5-qtgamepad-devel qt5-qtlocation-devel qt5-qtmultimedia-devel qt5-qtnetworkauth-devel qt5-qtquickcontrols2-devel qt5-qtremoteobjects-devel qt5-qtscript-devel qt5-qtscxml-devel qt5-qtsensors-devel qt5-qtserialbus-devel qt5-qtserialport-devel qt5-qtspeech-devel qt5-qtsvg-devel qt5-qttools-devel qt5-qtvirtualkeyboard-devel qt5-qtwayland-devel qt5-qtwebchannel-devel qt5-qtwebengine-devel qt5-qtwebkit-devel qt5-qtwebsockets-devel qt5-qtwebview-devel qt5-qtx11extras-devel qt5-qtxmlpatterns-devel qt5pas-devel qt5-qtaccountsservice-devel


```

## compile with qt

```
pkg-config Qt5Gui Qt5Widgets --cflags --libs

-I/usr/include/qt5/QtGui -I/usr/include/qt5 -I/usr/include/qt5/QtCore -DQT_WIDGETS_LIB -I/usr/include/qt5/QtWidgets -DQT_GUI_LIB -DQT_CORE_LIB


```

- compile2.sh can compile and link the qt example program within 1sec.

- it is important not to use spdlog. if this header is present in the
  module every recompilation takes >9sec