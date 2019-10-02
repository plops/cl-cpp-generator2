 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void cleanupSurface (){
        vkDestroySurfaceKHR(state._instance, state._surface, NULL);
}
void createSurface (){
        // initialize _surface member
        // must be destroyed before the instance is destroyed
        if ( !((VK_SUCCESS)==(glfwCreateWindowSurface(state._instance, state._window, NULL, &(state._surface)))) ) {
                        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" failed to (glfwCreateWindowSurface (dot state _instance) (dot state _window)            NULL (ref (dot state _surface))): ");
            printf("\n");
};
};
};