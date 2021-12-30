#/home/martin/src/filament/bin/cmgen \
# -x \
# --quiet --format=ktx --size=256 --extract-blur=0.1 \


mkdir generated
/home/martin/src/filament/bin/matc \
     --api all \
     -f header \
     -o generated/bakedColor.inc \
     bakedColor.mat
 
clang++ star_tracker.cpp \
	-std=c++17 \
	-I/home/martin/src/b/filament-1.15.1/out/cmake-release/libs/filamentapp/include/ \
	-I/home/martin/src/b/filament-1.15.1/libs/filamentapp/include/ \
	-I/home/martin/src/b/filament-1.15.1/out/release/filament/include \
	-I/home/martin/src/b/filament-1.15.1/third_party/libsdl2/include \
	-Wno-extern-c-compat \
	-ffast-math
 
