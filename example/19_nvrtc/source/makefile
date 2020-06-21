CXXFLAGS=-O0 -march=native -Wall -std=gnu++20 \
-I/media/sdb4/cuda/11.0.1/include/ \
-DNVRTC_GET_TYPE_NAME=1

LDFLAGS= \
-L/media/sdb4/cuda/11.0.1/lib \
-lcudart \
-lcuda \
-lnvrtc

OFILES = vis_00_main.o vis_01_rtc.o vis_02_cu_device.o

%.o: %.cpp globals.h proto2.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OFILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o vis vis_00_main.cpp

clean:
	rm vis
