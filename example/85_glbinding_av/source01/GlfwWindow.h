#ifndef GLFWWINDOW_H
#define GLFWWINDOW_H

class GLFWwindow; 
typedef void (*GLFWglproc)(void); 
class GlfwWindow  {
        GLFWwindow* m_window = nullptr; 
        public:
        explicit  GlfwWindow ()       ;   
         ~GlfwWindow ()       ;   
        bool WindowShouldClose ()       ;   
        void SwapBuffers ()       ;   
        GLFWwindow* GetWindow ()       ;   
        void PollEvents ()       ;   
        static GLFWglproc GetProcAddress (const char* name)       ;   
        std::pair<int,int> GetWindowSize () const      ;   
};

#endif /* !GLFWWINDOW_H */