#pragma once
#include <vector>
#include <functional> 
class GLFWwindow; 
class ImVec4; 
class ImGuiIO; 
class MainWindow  {
        public:
        bool show_demo_window_; 
        ImGuiIO& io; 
        explicit  MainWindow ()       ;   
         ~MainWindow ()       ;   
        void Init (GLFWwindow* window, const char* glsl_version)       ;   
        void NewFrame ()       ;   
        void Update (std::function<void(void)> fun)       ;   
        void Render (GLFWwindow* window)       ;   
        void Shutdown ()       ;   
};
