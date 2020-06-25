CXXFLAGS=-O0 -march=native -Wall -std=gnu++17 \
-I/media/sdb4/cuda/11.0.1/include/ \
-DNVRTC_GET_TYPE_NAME=1

LDFLAGS= \
-L/media/sdb4/cuda/11.0.1/lib \
-lcudart \
-lcuda \
-lnvrtc \
-Wl,-rpath=/media/sdb4/cuda/11.0.1/lib

OFILES = vis_00_main.o vis_01_cu_A_rtc_code.o vis_02_cu_A_device.o vis_03_cu_A_context.o vis_04_cu_A_rtc_compilation_options.o vis_05_cu_A_rtc_header.o vis_06_cu_A_rtc_program.o vis_07_cu_A_rtc_kernel.o vis_08_cu_A_rtc_module.o 

%.o: %.cpp globals.h proto2.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OFILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm vis $(OFILES)
