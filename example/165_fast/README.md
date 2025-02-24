# Intro

A cache friendly storage solution

# Files

|       |                            |
| gen01 | stable vector              |
| gen02 | random walk through memory |
|       |                            |


# Dependencies:

Google Benchmark
stable_vector (also requires boost)

```
cd ~/src
git clone https://github.com/google/benchmark
git clone https://github.com/david-grs/stable_vector
```

## Papi

```
cd ~/src
wget -c https://icl.utk.edu/projects/papi/downloads/papi-7.2.0b1.tar.gz
tar xaf papi-7.2.0b1.tar.gz
cd papi-7.2.0b1/src
./configure --prefix=/home/martin/vulkan
make -j12
make install
cd ~/src
git clone https://github.com/david-grs/papipp
cd papipp
mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/martin/vulkan
```



# References:

## Trading at light speed - designing low latency systems in C++ - David Gross - Meeting C++ 2022
8uAW5FQtcvE


## When_Nanoseconds_Matter_-_Ultrafast_Trading_Systems_in_C++_-_David_Gross_-_C24-ppCon_2024-[sX2nF1fW7kI].webm



00:27:44 running perf without initialization
intel tdmaam

top down microarchitacture analysis method

30:51 perf record -g -p <pid>
33:00 hw counters libpapi

37:00 [[likely]]

50:00 fast queue write counter and read countre

1:00:00 xray clang (nops at every function can be patched with function calls during profiling)

1:11:00 share l3 cache with others (you need to think about the system as a whole)
