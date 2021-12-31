all: star_tracker
generated/materials/bakedColor.filamat: bakedColor.mat
	/home/martin/src/filament/bin/matc --api all \
     -o generated/materials/bakedColor.filamat \
     bakedColor.mat

generated/resources.S: generated/materials/bakedColor.filamat
	/home/martin/src/filament/bin/resgen \
	--deploy=generated \
	generated/materials/bakedColor.filamat
CXX=clang++
CXXFLAGS=-std=c++17 -flto \
	-I/home/martin/src/b/filament-1.15.1/libs/filamentapp/include/ \
	-I/home/martin/src/b/filament-1.15.1/out/release/filament/include \
	-I/home/martin/src/b/filament-1.15.1/third_party/libsdl2/include \
	-I/home/martin/src/b/filament-1.15.1/third_party/imgui \
	-Wno-extern-c-compat \
	-ffast-math \
	-fstrict-aliasing -Wno-unknown-pragmas -Wno-unused-function -fPIC \
	-fcolor-diagnostics -fvisibility=hidden -ggdb -O3 -DNDEBUG \
	-fomit-frame-pointer -ffunction-sections -fdata-sections

# -L/home/martin/src/b/filament-1.15.1/out/cmake-release/ 
LDFLAGS=-Wl,--gc-sections \
	libs/filamentapp/libfilamentapp.a \
	third_party/libassimp/tnt/libassimp.a \
	third_party/libz/tnt/libz.a  libs/iblprefilter/libfilament-iblprefilter.a  \
	third_party/libsdl2/tnt/libsdl2.a  libs/filamat/libfilamat.a  shaders/libshaders.a \
	-Wl,--start-group  third_party/glslang/tnt/SPIRV/libSPIRV.a  \
	third_party/glslang/tnt/glslang/libglslang.a  third_party/glslang/tnt/OGLCompilersDLL/libOGLCompiler.a  \
	third_party/glslang/tnt/glslang/OSDependent/Unix/libOSDependent.a  \
	third_party/glslang/tnt/SPIRV/libSPVRemapper.a  third_party/spirv-tools/source/opt/libSPIRV-Tools-opt.a \
	third_party/spirv-tools/source/libSPIRV-Tools.a  /usr/lib64/librt.a \
	third_party/spirv-cross/tnt/libspirv-cross-glsl.a  third_party/spirv-cross/tnt/libspirv-cross-msl.a  \
	third_party/spirv-cross/tnt/libspirv-cross-core.a \
	-Wl,--end-group  third_party/getopt/libgetopt.a  libs/filagui/libfilagui.a  filament/libfilament.a \
	filament/backend/libbackend.a  libs/bluegl/libbluegl.a  libs/bluevk/libbluevk.a  \
	filament/backend/libvkshaders.a  libs/filaflat/libfilaflat.a  third_party/smol-v/tnt/libsmol-v.a \
	libs/filabridge/libfilabridge.a  libs/ibl/libibl-lite.a  -Wl,--exclude-libs,bluegl  \
	third_party/imgui/tnt/libimgui.a  libs/image/libimage.a  libs/camutils/libcamutils.a  \
	libs/geometry/libgeometry.a  libs/math/libmath.a  libs/utils/libutils.a \
	-ldl  libs/filamentapp/libfilamentapp-resources.a 


star_tracker: star_tracker.cpp generated/resources.S 
	cd generated; ${CXX} ${CXXFLAGS} \
	resources.S \
	../star_tracker.cpp \
	-o ../star_tracker ${LDFLAGS}

clean:
	rm generated/materials/bakedColor.filamat
	rm generated/resources.{bin,S,apple.S,h}
	rm star_tracker
