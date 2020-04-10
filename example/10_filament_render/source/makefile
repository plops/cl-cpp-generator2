FILAMENT_LIBS=-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -libl
CC=clang++

all: render_00_main

CFILES = /home/martin/src/filament/libs/filamentapp/src/Cube.cpp \
/home/martin/src/filament/libs/filamentapp/src/NativeWindowHelperLinux.cpp \
/home/martin/src/filament/libs/filamentapp/src/IBL.cpp \
/home/martin/src/filament/libs/filamentapp/src/NativeWindowHelperWindows.cpp \
/home/martin/src/filament/libs/filamentapp/src/Sphere.cpp \
/home/martin/src/filament/libs/filamentapp/src/Image.cpp \
/home/martin/src/filament/libs/filamentapp/src/MeshAssimp.cpp \
/home/martin/src/filament/libs/filamentapp/src/FilamentApp.cpp \
/home/martin/src/filament/libs/filamentapp/src/IcoSphere.cpp

OBJ := $(notdir $(CFILES:%.cpp=%.o))

CXXFLAGS=-Ifilament/include/ -I/home/martin/src/filament/libs/filamentapp/include -I/usr/include/SDL2 

render_00_main: render_00_main.o $(OBJ)
	$(CC)  $(CXXFLAGS) -Lfilament/lib/x86_64/ $< $(FILAMENT_LIBS) -lpthread -lc++ -ldl -o $@

render_00_main.o: render_00_main.cpp
	$(CC)  $(CXXFLAGS) -std=c++14 -pthread -c render_00_main.cpp

%.o: %.cpp
	$(CC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f render_00_main.o render_00_main

.PHONY: clean
