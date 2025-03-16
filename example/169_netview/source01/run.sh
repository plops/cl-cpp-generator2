#!/bin/bash


ln -s cmake-build-release/cxxnet server
ln -s cmake-build-release/cxxnet client

cd cmake-build-release/
ninja
cd ..

killall server

./server localhost:43211 &
./client localhost:43211
