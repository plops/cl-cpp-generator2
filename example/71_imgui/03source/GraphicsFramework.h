#pragma once
#include <memory> 
class GLFWwindow; 
class GraphicsFramework  {
        public:
        std::shared_ptr<GLFWwindow> window; 
         GraphicsFramework ()       ;   
         ~GraphicsFramework ()       ;   
        bool WindowShouldClose ()       ;   
        void PollEvents ()       ;   
        std::shared_ptr<GLFWwindow> getWindow ()       ;   
};
