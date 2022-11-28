cmake_minimum_required( VERSION 3.4 )
project( mytest LANGUAGES CXX )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused ")
find_package( Qt5 COMPONENTS Core Gui PrintSupport Widgets Charts REQUIRED )
set( SRCS /home/martin/stage/cl-cpp-generator2/example/70_qttest/source/CpuWidget.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/MainWindow.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/MemoryWidget.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/SysInfo.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/SysInfoLinuxImpl.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/SysInfoWidget.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/main.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/moc_CpuWidget.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/moc_MainWindow.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/moc_MemoryWidget.cpp
/home/martin/stage/cl-cpp-generator2/example/70_qttest/source/moc_SysInfoWidget.cpp )
add_executable( mytest ${SRCS} )
target_compile_features( mytest PUBLIC cxx_std_17 )
find_package ( PkgConfig REQUIRED )
pkg_check_modules( QCP REQUIRED qcustomplot-qt5 )
target_link_libraries( mytest PRIVATE Qt::Core Qt::Gui Qt::PrintSupport Qt::Widgets Qt::Charts qcustomplot-qt5 )
