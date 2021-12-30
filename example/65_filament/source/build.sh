/home/martin/src/filament/bin/cmgen \
 -x \
 --quiet --format=ktx --size=256 --extract-blur=0.1 \


mkdir generated
/home/martin/src/filament/bin/matc \
     --api all \
     -f header \
     -o generated/bakedColor.inc \
     bakedColor.mat
 
clang++ star_tracker.cpp \
	-std=c++17 \
	-I/home/martin/src/filament/include \
	-Wno-extern-c-compat \
	-ffast-math
 
