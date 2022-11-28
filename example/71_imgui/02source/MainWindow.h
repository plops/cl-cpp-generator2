#pragma once
#include <vector>
#include <functional>
#include <memory>
class GLFWwindow;
class ImVec4;
class ImGuiIO;
class MainWindow  {
        public:
        bool show_demo_window_;
        ImGuiIO& io;
        explicit  MainWindow ()     ;  
         ~MainWindow ()     ;  
        void Init (std::shared_ptr< GLFWwindow > window, const char* glsl_version)     ;  
        void NewFrame ()     ;  
        void Update (std::function<void(void)> fun)     ;  
        void Render (std::shared_ptr< GLFWwindow > window)     ;  
        void Shutdown ()     ;  
};
