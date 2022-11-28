cat main.cpp | sed 's/^[ \t]*//' > main_trim.c
./call_python.sh /home/martin/stage/cl-cpp-generator2/example/02_shader/source01/main_trim.c
