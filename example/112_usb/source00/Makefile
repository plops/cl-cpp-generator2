all: main

UsbError.o: UsbError.cpp 
	clang++ -std=c++20 -ggdb -O1 UsbError.cpp -c -o UsbError.o


main.o: main.cpp 
	clang++ -std=c++20 -ggdb -O1 main.cpp -c -o main.o

main: main.o UsbError.o UsbInterface.hpp Usbpp.hpp
	clang++ -std=c++20 -ggdb -O1 main.o UsbError.o -o main `pkg-config libusb-1.0 --libs`

distclean: 
	rm fatheader.pcm UsbInterface.o UsbError.o main.o main 

clean:
	rm UsbInterface.o UsbError.o main.o main
