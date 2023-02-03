time clang++ -std=c++20 -ggdb -O1 -fmodule-file=fatheader.pcm main.cpp -c -o main.o
time clang++ -std=c++20 -ggdb -O1 -fmodule-file=fatheader.pcm UsbError.cpp -c -o UsbError.o
time clang++ -std=c++20 -ggdb -O1 main.o UsbError.o -o main `pkg-config libusb-1.0 --libs`
