
for i in implot.h implot_internal.h implot.cpp implot_items.cpp; do
    wget https://raw.githubusercontent.com/epezent/implot/master/$i
done

for i in \
    imconfig.h \
	imgui.h imgui.cpp imgui_draw.cpp imgui_internal.h imgui_tables.cpp \
	imgui_widgets.cpp imstb_rectpack.h imstb_textedit.h imstb_truetype.h \
	backends/imgui_impl_glfw.h backends/imgui_impl_opengl3.h \
	backends/imgui_impl_opengl3.cpp backends/imgui_impl_opengl3_loader.h \
	backends/imgui_impl_glfw.cpp \
	misc/cpp/imgui_stdlib.cpp misc/cpp/imgui_stdlib.h ;do
    wget  https://raw.githubusercontent.com/ocornut/imgui/master/$i
done

