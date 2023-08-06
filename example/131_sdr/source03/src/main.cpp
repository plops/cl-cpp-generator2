#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <array>
#include <complex>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility> 
#include "implot.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "GLFW/glfw3.h" 

void glfw_error_callback (int err, const char* desc)        {
        
}
 
auto DrawPlot  = [] (){
        try {
                        ImGui::Text("hello");
 
}catch (const std::exception& e) {
                ImGui::Text("Error while processing signal: %s", e.what());
} 
}; 
 
auto initGL  = [] (){
            glfwSetErrorCallback(glfw_error_callback);
    if ( 0==glfwInit() ) {
                        
 
} 
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
 
        auto *window  = glfwCreateWindow(800, 600, "imgui_dsp", nullptr, nullptr); 
    if ( nullptr==window ) {
                        
 
} 
    glfwMakeContextCurrent(window);
    
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    
    ImGui::CreateContext();
    ImPlot::CreateContext();
        auto &io  = ImGui::GetIO(); 
        io.ConfigFlags=io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;


 
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    glClearColor(0, 0, 0, 1);
 
 
        return window;
}; 
 

int main (int argc, char** argv)        {
        try {
                                auto *window  = initGL(); 
 
        while ( !glfwWindowShouldClose(window) ) {
                        glfwPollEvents();
                        ImGui_ImplOpenGL3_NewFrame();
                        ImGui_ImplGlfw_NewFrame();
                        ImGui::NewFrame();
                        DrawPlot();
                        ImGui::Render();
                                    auto w  = 0; 
            auto h  = 0; 
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);
            glClear(GL_COLOR_BUFFER_BIT);
 
                        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                        glfwSwapBuffers(window);
} 
                ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
 
 
}catch (const std::runtime_error& e) {
                
                return -1;
}catch (const std::exception& e) {
                
                return -1;
} 
        return 0;
}
 
