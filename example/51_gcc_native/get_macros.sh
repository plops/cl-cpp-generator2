#!/bin/bash
echo -n "// "
cat /proc/cpuinfo |grep "model name"|uniq 

echo -n "// "
gcc -v 2>&1|grep "gcc version"

gcc -march=native -dM -E - < /dev/null 
