mkdir -p /home/martin/stage/cl-cpp-generator2/example/71_imgui/01source/b
cd /home/martin/stage/cl-cpp-generator2/example/71_imgui/01source/
cmake -B b -S . -DCMAKE_TOOLCHAIN_FILE=/home/martin/src/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug -G Ninja
cmake --build b
/home/martin/stage/cl-cpp-generator2/example/71_imgui/01source/b/mytest
