clang-tidy \
    main.cpp \
    -checks=-*,cppcoreguidelines-*,clang-analyzer-*,-clang-analyzer-cplusplus* \
    -- \
    -I/home/martin/src/bgfx/include \
    -I/home/martin/src/bx/include \
    -I/home/martin/src/bgfx/3rdparty/ \
    -I/home/martin/src/bimg/include/ \
    -DBX_CONFIG_DEBUG
    
