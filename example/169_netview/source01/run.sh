#!/bin/bash

ln -s cmake-build-release/cxxnet server
ln -s cmake-build-release/cxxnet client

killall server

./server localhost:43211 &
./client localhost:43211
