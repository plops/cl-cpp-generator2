#pragma once
#include <vector> 
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
        void Update ()       ;   
        void Render (GLFWwindow* window)       ;   
        void Shutdown ()       ;   
};
