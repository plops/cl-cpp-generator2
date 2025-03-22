#!/bin/bash
BUILD_DIR=../source01/cmake-build-debug
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake .. -G Ninja \
-DCMAKE_BUILD_TYPE=Debug \
-DENABLE_ASAN:BOOL=ON -DENABLE_UBSAN=ON -DENABLE_LSAN=ON \
-DENABLE_VID_TESTS:BOOL=ON
ninja &&
./unit_tests &&
mkdir -p gcov &&
cd gcov &&
find ..|grep gcda |xargs \
gcov --human-readable \
--demangled-names \
--use-colors