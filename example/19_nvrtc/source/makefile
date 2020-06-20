CXXFLAGS=-O0 -march=native -Wall -std=gnu++20 \
-I/media/sdb4/cuda/11.0.1/include/
LDFLAGS= \
-L/media/sdb4/cuda/11.0.1/lib \
-lcudart \
-lcuda \
-lnvrtc

vis: vis_00_main.cpp vis_01_rtc.cpp vis_02_cu_device.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o vis vis_00_main.cpp

clean:
	rm vis
