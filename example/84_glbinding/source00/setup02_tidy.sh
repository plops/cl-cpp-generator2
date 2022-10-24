clang-tidy \
    main.cpp \
    -checks=-*,cppcoreguidelines-*,clang-analyzer-*,-clang-analyzer-cplusplus* \
    -extra-arg=-std=c++20 \
    -- \
    -I/usr/local/include \
    -I/home/martin/src/imgui

   
    
