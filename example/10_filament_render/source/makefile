FILAMENT_LIBS=-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -libl
CC=clang++

all: render_00_main

render_00_main: render_00_main.o
	$(CC) -Lfilament/lib/x86_64/ render_00_main.o $(FILAMENT_LIBS) -lpthread -lc++ -ldl -o render_00_main

render_00_main.o: render_00_main.cpp
	$(CC) -Ifilament/include/ -std=c++14 -pthread -c render_00_main.cpp

clean:
	rm -f render_00_main.o render_00_main

.PHONY: clean
