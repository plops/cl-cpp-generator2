#!/bin/bash

rm -rf server client
ln -s cmake-build-release/cxxnet_client server
ln -s cmake-build-release/cxxnet_client client

cd cmake-build-release/
ninja
cd ..

killall server

./server localhost:43211 &
./client localhost:43211
