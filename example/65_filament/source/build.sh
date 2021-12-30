# cmgen is for environment maps
#/home/martin/src/filament/bin/cmgen \
# -x \
# --quiet --format=ktx --size=256 --extract-blur=0.1 \





mkdir -p generated/materials
/home/martin/src/filament/bin/matc \
     --api all \
     -o generated/materials/bakedColor.filamat \
     bakedColor.mat

# https://github.com/google/filament/blob/main/ios/samples/hello-pbr/build-resources.sh
/home/martin/src/filament/bin/resgen \
 --deploy=generated \
 generated/materials/bakedColor.filamat

cd generated
ln -s /home/martin/src/b/filament-1.15.1/out/cmake-release/third_party .
ln -s /home/martin/src/b/filament-1.15.1/out/cmake-release/libs .
ln -s /home/martin/src/b/filament-1.15.1/out/cmake-release/filament .
ln -s /home/martin/src/b/filament-1.15.1/out/cmake-release/shaders .
clang++ ../star_tracker.cpp \
	-std=c++17 -flto \
	-I/home/martin/src/b/filament-1.15.1/out/cmake-release/libs/filamentapp/include/ \
	-I/home/martin/src/b/filament-1.15.1/libs/filamentapp/include/ \
	-I/home/martin/src/b/filament-1.15.1/out/release/filament/include \
	-I/home/martin/src/b/filament-1.15.1/third_party/libsdl2/include \
	-Wno-extern-c-compat \
	-ffast-math \
	resources.S \
	-L/home/martin/src/b/filament-1.15.1/out/release/filament/lib/x86_64 \
	-lbluevk -lfilament -lutils -lvkshaders -lbackend \
	-L/home/martin/src/b/filament-1.15.1/out/cmake-release/libs/filamentapp \
	-lfilamentapp \
	-L/home/martin/src/b/filament-1.15.1/out/cmake-release/third_party/libsdl2/tnt \
	-lsdl2 \
	-fstrict-aliasing -Wno-unknown-pragmas -Wno-unused-function -fPIC \
	-fcolor-diagnostics -fvisibility=hidden -O3 -DNDEBUG \
	-fomit-frame-pointer -ffunction-sections -fdata-sections \
	-Wl,--gc-sections \
	-L/home/martin/src/b/filament-1.15.1/out/cmake-release/ libs/filamentapp/libfilamentapp.a \
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
mv a.out ..
