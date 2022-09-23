#!/bin/bash

# print the commands
set -x

# run tla simulation
java -XX:+UseParallelGC \
     -jar ./tla2tools.jar \
     -deadlock -dump dot p1c1b1.dot queue 

# convert graph to pdf
dot -Tpdf p1c1b1.dot -o p1c1b1.pdf
