#!/bin/bash

rm -rf server client

REL=cmake-build-debug-nolibs
ln -s $REL/cxxnet_client server
ln -s $REL/cxxnet_client client

cd $REL/
ninja
cd ..

killall server

./server localhost:43211 &
./client localhost:43211
