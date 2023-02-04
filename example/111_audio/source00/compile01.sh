#!/bin/bash
clang++ -o main main.cpp \
	`pkg-config libpipewire-0.3 --libs --cflags` \
	`pkg-config fmt --libs --cflags`
