#!/usr/bin/bash

if [ ! -f svdconv ]; then
    wget https://github.com/Open-CMSIS-Pack/devtools/releases/download/tools%2Fsvdconv%2F3.3.46/svdconv-3.3.46-linux-amd64.tbz2
    tar xaf svdconv-3.3.46-linux-amd64.tbz2
fi


./svdconv --generate=header --strict --fields=struct-ansic  ../CH59Xxx.svd
