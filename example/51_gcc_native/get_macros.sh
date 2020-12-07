#!/bin/bash
echo -n "# "
cat /proc/cpuinfo |grep "model name"|uniq 
gcc -march=native -dM -E - < /dev/null 
