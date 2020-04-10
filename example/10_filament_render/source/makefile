FILAMENT_LIBS=-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -libl
CC=clang++

main: main.o
        $(CC) -Lfilament/lib/x86_64/ main.o $(FILAMENT_LIBS) -lpthread -lc++ -ldl -o main

main.o: main.cpp
        $(CC) -Ifilament/include/ -std=c++14 -pthread -c main.cpp

clean:
        rm -f main main.o

.PHONY: clean
