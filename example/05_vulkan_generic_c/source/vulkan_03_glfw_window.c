 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
static void framebufferResizeCallback (GLFWwindow* window, int width, int height){
            __auto_type app  = (State*)(glfwGetWindowUserPointer(window));
        app->_framebufferResized=true;
}
void initWindow (){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            state._window=glfwCreateWindow(800, 600, "vulkan window", NULL, NULL);
        glfwSetWindowUserPointer(state._window, &(state));
        glfwSetFramebufferSizeCallback(state._window, framebufferResizeCallback);
}
void cleanupWindow (){
        glfwDestroyWindow(state._window);
        glfwTerminate();
};