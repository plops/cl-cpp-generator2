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

cmake /home/martin/stage/cl-cpp-generator2/example/95_vr/source00
