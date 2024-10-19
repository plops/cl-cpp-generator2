#!/usr/bin/env bash

export MY_EXTRA_FLAGS='-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/deps143 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_C_FLAGS="-march=native"'

export DEPS_SRC_DIR=/mnt5/tmp/src
cd $DEPS_SRC_DIR


export TARGET=glfw
export BUILD_DIR="/tmp/build_$TARGET"
export SRC_DIR=$DEPS_SRC_DIR/$TARGET
mkdir -p $BUILD_DIR

echo "Building $TARGET ..."
cmake $MY_EXTRA_FLAGS -S $SRC_DIR -B $BUILD_DIR \
 -DGLFW_BUILD_DOCS=OFF  \
 -DGLFW_BUILD_EXAMPLES=OFF \
 -DGLFW_BUILD_TESTS=OFF     \
 -DGLFW_BUILD_WAYLAND=OFF    \
 -DGLFW_BUILD_X11=ON          \
 -DGLFW_INSTALL=ON
cmake --build $BUILD_DIR
cmake --install $BUILD_DIR


export TARGET=popl
export BUILD_DIR="/tmp/build_$TARGET"
export SRC_DIR=$DEPS_SRC_DIR/$TARGET
mkdir -p $BUILD_DIR

echo "Building $TARGET ..."
cmake $MY_EXTRA_FLAGS -S $SRC_DIR -B $BUILD_DIR \
 -DBUILD_TESTS=OFF \
 -DBUILD_EXAMPLE=OFF
cmake --build $BUILD_DIR
cmake --install $BUILD_DIR


# Glew should be installed from release tar.gz
# the git repo is complicated to compile because it contains
# a code-generation step and requires sub repos

#export TARGET=glew
#export BUILD_DIR="/tmp/build_$TARGET"
#export SRC_DIR=$DEPS_SRC_DIR/$TARGET/build/cmake
#mkdir -p $BUILD_DIR
#
#echo "Building $TARGET ..."
#cmake $MY_EXTRA_FLAGS -S $SRC_DIR -B $BUILD_DIR \
# -DBUILD_TESTS=OFF \
# -DBUILD_EXAMPLE=OFF
#cmake --build $BUILD_DIR
#cmake --install $BUILD_DIR