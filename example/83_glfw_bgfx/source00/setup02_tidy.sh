clang-tidy \
    main.cpp \
    -checks=-*,cppcoreguidelines-*,clang-analyzer-*,-clang-analyzer-cplusplus* \
    -extra-arg=-std=c++20 \
    -- \
    -DBX_CONFIG_DEBUG \
    -I/home/martin/src/bgfx/include \
    -I/home/martin/src/bx/include \
    -I/home/martin/src/bgfx/3rdparty/ \
    -I/home/martin/src/bimg/include/ \
    -I/home/martin/src/bgfx/examples/common \
    -I/home/martin/src/entt/src/ 
   
    
