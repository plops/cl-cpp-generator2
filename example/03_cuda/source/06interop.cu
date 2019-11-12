// nvcc -o 06interop 06interop.cu -lglfw
// note that nvcc requires gcc 8
// nvprof 06interpo
#include <GLFW/glfw3.h>
int main (){
        if ( !(glfwInit()) ) {
                        glfwTerminate();
};
}