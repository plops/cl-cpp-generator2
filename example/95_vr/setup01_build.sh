#!/bin/bash
export PATH=$PATH:/home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin
export PATH=$PATH:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.352.b08-2.fc37.x86_64/bin/
export PATH=$PATH:/home/martin/quest2/sdk/build-tools/29.0.3

rm -rf build
mkdir -p build
pushd build

javac \
	-classpath /home/martin/quest2/sdk/platforms/android-29/android.jar \
	-d . \
	/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/java/MainActivity.java

# output: build/com/makepad/hello_quest/MainActivity.class

dx --dex --output classes.dex .

# output: build/classes.dex

mkdir -p lib/arm64-v8a
pushd lib/arm64-v8a

export CC=aarch64-linux-android29-clang
export CXX=aarch64-linux-android29-clang++
cmake /home/martin/stage/cl-cpp-generator2/example/95_vr/source00
make -j4


cp /home/martin/quest2/ovr/VrApi/Libs/Android/arm64-v8a/Debug/libvrapi.so .
cp /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so .
cp /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/29/libEGL.so .

popd

aapt package \
     -F hello_quest.apk \
     -I /home/martin/quest2/sdk/platforms/android-29/android.jar \
     -M /home/martin/stage/cl-cpp-generator2/example/95_vr/source00/AndroidManifest.xml  \
     -f 

# creates hello_quest.apk

aapt add hello_quest.apk classes.dex
aapt add hello_quest.apk lib/arm64-v8a/libmain.so
aapt add hello_quest.apk lib/arm64-v8a/libvrapi.so
aapt add hello_quest.apk lib/arm64-v8a/libc++_shared.so
aapt add hello_quest.apk lib/arm64-v8a/libEGL.so

apksigner sign \
	  -ks ~/.android/debug.keystore \
	  --ks-key-alias androiddebugkey \
	  --ks-pass pass:android \
	  hello_quest.apk

popd
