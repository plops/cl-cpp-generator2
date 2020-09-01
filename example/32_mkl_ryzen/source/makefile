all: libfake_intel.so dgemm

libfake_intel.so: vis_00_fake_intel.cpp
	g++ -shared -fPIC -O2 -march=native -o $@ $^

#g++ `pkg-config mkl-static-lp64-seq --cflags --libs` -o $@ $^ -fopenmp

CXXFLAGS= \
-march=native \
-O2 \
-fopenmp \
-pipe \
-I/media/sdb4/programs/intel/compilers_and_libraries_2020.2.254/linux/mkl/include

LDFLAGS= -L/media/sdb4/programs/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/ \
-lmkl_core -lmkl_gnu_thread -lmkl_intel_lp64 \
-lpthread -lm -ldl -Wl,-rpath /media/sdb4/programs/intel/compilers_and_libraries_2020.2.254/linux/mkl/lib/intel64_lin/


dgemm: vis_01_base.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^
