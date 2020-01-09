for i in imconfig.h imgui.cpp imgui.h imgui_demo.cpp imgui_draw.cpp examples/imgui_impl_glfw.cpp examples/imgui_impl_glfw.h examples/imgui_impl_opengl2.cpp examples/imgui_impl_opengl2.h imgui_internal.h imgui_widgets.cpp imstb_rectpack.h imstb_textedit.h imstb_truetype.h; do
    wget https://raw.githubusercontent.com/ocornut/imgui/master/$i
done
