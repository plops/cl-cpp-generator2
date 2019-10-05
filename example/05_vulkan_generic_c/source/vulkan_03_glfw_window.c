 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
static void framebufferResizeCallback (GLFWwindow* window, int width, int height){
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" resize: ");
        printf(" width=");
        printf(printf_dec_format(width), width);
        printf(" (%s)", type_string(width));
        printf(" height=");
        printf(printf_dec_format(height), height);
        printf(" (%s)", type_string(height));
        printf("\n");
};
            __auto_type app  = (State*)(glfwGetWindowUserPointer(window));
        app->_framebufferResized=true;
}
void initWindow (){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
            state._window=glfwCreateWindow(800, 600, "vulkan window", NULL, NULL);
        glfwSetWindowUserPointer(state._window, &(state));
        glfwSetFramebufferSizeCallback(state._window, framebufferResizeCallback);
}
void cleanupWindow (){
        glfwDestroyWindow(state._window);
        glfwTerminate();
};