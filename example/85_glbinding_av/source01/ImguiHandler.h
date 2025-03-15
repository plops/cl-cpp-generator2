#ifndef IMGUIHANDLER_H
#define IMGUIHANDLER_H

class GLFWwindow; 
class ImguiHandler  {
        public:
        explicit  ImguiHandler (GLFWwindow* window)       ;   
        void NewFrame ()       ;   
        void Render ()       ;   
        void RenderDrawData ()       ;   
        void Begin (const char* str)       ;   
        void End ()       ;   
        void Image (uint tex, int w, int h)       ;   
        void SliderFloat (const char* label, float* val, float min, float max, const char* fmt)       ;   
         ~ImguiHandler ()       ;   
};

#endif /* !IMGUIHANDLER_H */