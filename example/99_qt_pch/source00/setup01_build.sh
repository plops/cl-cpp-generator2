#!/bin/bash

mkdir b
cd b
CC=clang-14 CXX=clang+++-14 cmake ..
make -j4
