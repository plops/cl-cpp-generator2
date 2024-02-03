mkdir b
cd b
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_RYZEN_TESTS=ON .. -DDEP_DIR=/home/martin/src
