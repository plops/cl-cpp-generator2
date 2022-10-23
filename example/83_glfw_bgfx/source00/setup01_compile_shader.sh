cd /home/martin/stage/cl-cpp-generator2/example/83_glfw_bgfx/source00

/home/martin/src/bgfx/.build/linux64_gcc/bin/shadercRelease \
 -f v_simple.sc \
 -o v_simple.bin \
 --platform linux \
 --type vertex \
 -p spirv \
 --verbose \
 -i /home/martin/src/bgfx/src

/home/martin/src/bgfx/.build/linux64_gcc/bin/shadercRelease \
 -f f_simple.sc \
 -o f_simple.bin \
 --platform linux \
 -p spirv \
 --type fragment \
 --verbose \
 -i /home/martin/src/bgfx/src

cp *.bin b
