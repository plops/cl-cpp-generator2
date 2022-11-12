#ifndef GLFWWINDOW_H
#define GLFWWINDOW_H

class GLFWwindow;
class GlfwWindow  {
        GLFWwindow* m_window = nullptr;
        public:
        explicit  GlfwWindow ()     ;  
         ~GlfwWindow ()     ;  
        bool WindowShouldClose ()     ;  
        void SwapBuffers ()     ;  
        GLFWwindow* GetWindow ()     ;  
        std::pair<int,int> GetWindowSize () const    ;  
};

#endif /* !GLFWWINDOW_H */