#!/bin/bash

ln -s cmake-build-release/cxxnet server
ln -s cmake-build-release/cxxnet client

./server &
./client 43211
