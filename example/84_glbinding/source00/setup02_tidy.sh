clang-tidy \
    main.cpp \
    -checks=-*,cppcoreguidelines-*,clang-analyzer-*,-clang-analyzer-cplusplus* \
    -extra-arg=-std=c++20 \
    -- \
    -I/usr/local/include \
    -I/home/martin/src/imgui \
    -I/home/martin/src/entt/src/ \
    -I/home/martin/src/imgui_entt_entity_editor/

   
    
