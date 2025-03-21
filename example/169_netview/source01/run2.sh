#!/bin/bash

rm -rf server client

REL=cmake-build-release-nolibs
SERVER=10.0.0.203

cd $REL/
ninja
cd ..

killall ssh
rsync $REL/cxxnet_client pop-os:/dev/shm/server
ssh pop-os /dev/shm/server &

$REL/cxxnet_client $SERVER:43211

