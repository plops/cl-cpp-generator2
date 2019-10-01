 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <string.h>
double now (){
            struct timespec tp ;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (((((1.e+6f))*(tp.tv_sec)))+(tp.tv_nsec));
}
void updateUniformBuffer (uint32_t currentImage){
            auto double startTime  = now();
    __auto_type currentTime  = now();
    float time  = ((currentTime)-(startTime));
        __auto_type zAxis  = (vec3) {(0.0e+0f), (0.0e+0f), (1.e+0f)};
    __auto_type eye  = (vec3) {(2.e+0f), (2.e+0f), (2.e+0f)};
    __auto_type center  = (vec3) {(0.0e+0f), (0.0e+0f), (0.0e+0f)};
    __auto_type angularRate  = glm_rad((9.e+0f));
    __auto_type rotationAngle  = ((time)*(angularRate));
    mat4 identity ;
    mat4 model ;
    mat4 look ;
    mat4 projection ;
        glm_mat4_identity(identity);
    glm_rotate_z(identity, rotationAngle, model);
        glm_lookat(eye, center, zAxis, look);
        glm_perspective(glm_rad((4.5e+1f)), ((state._swapChainExtent.width)/((((1.e+0f))*(state._swapChainExtent.height)))), (1.e-1f), (1.e+1f), projection);
        UniformBufferObject ubo  = {};
    glm_mat4_copy(model, ubo.model);
    glm_mat4_copy(look, ubo.view);
    glm_mat4_copy(projection, ubo.proj);
        ubo.proj[1][1]=(-(ubo.proj[1][1]));
        void* data  = 0;
    vkMapMemory(state._device, state._uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(state._device, state._uniformBuffersMemory[currentImage]);
}
void recreateSwapChain (){
            int width  = 0;
    int height  = 0;
    while ((((0)==(width))||((0)==(height)))) {
                glfwGetFramebufferSize(state._window, &width, &height);
                {
                                    struct timespec tp ;
            clock_gettime(CLOCK_REALTIME, &tp);
            printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
            printf(".");
            printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" get frame buffer size: ");
            printf(" width=");
            printf(printf_dec_format(width), width);
            printf(" (%s)", type_string(width));
            printf(" height=");
            printf(printf_dec_format(height), height);
            printf(" (%s)", type_string(height));
            printf("\n");
};
                glfwWaitEvents();
};
        {
                        struct timespec tp ;
        clock_gettime(CLOCK_REALTIME, &tp);
        printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
        printf(".");
        printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" wait idle: ");
        printf("\n");
};
        vkDeviceWaitIdle(state._device);
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        {
                        struct timespec tp ;
        clock_gettime(CLOCK_REALTIME, &tp);
        printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
        printf(".");
        printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" swap chain has been recreated.: ");
        printf("\n");
};
}
void drawFrame (){
        {
                        struct timespec tp ;
        clock_gettime(CLOCK_REALTIME, &tp);
        printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
        printf(".");
        printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" wait for fences: ");
        printf(" (aref (dot state _inFlightFences) (dot state _currentFrame))=");
        printf(printf_dec_format(state._inFlightFences[state._currentFrame]), state._inFlightFences[state._currentFrame]);
        printf(" (%s)", type_string(state._inFlightFences[state._currentFrame]));
        printf(" (dot state _currentFrame)=");
        printf(printf_dec_format(state._currentFrame), state._currentFrame);
        printf(" (%s)", type_string(state._currentFrame));
        printf("\n");
};
            vkWaitForFences(state._device, 1, &(state._inFlightFences[state._currentFrame]), VK_TRUE, UINT64_MAX);
            uint32_t imageIndex  = 0;
    __auto_type result  = vkAcquireNextImageKHR(state._device, state._swapChain, UINT64_MAX, state._imageAvailableSemaphores[state._currentFrame], VK_NULL_HANDLE, &imageIndex);
    if ( (VK_ERROR_OUT_OF_DATE_KHR)==(result) ) {
                        recreateSwapChain();
        return ;
};
    if ( !((((VK_SUCCESS)==(result))||((VK_SUBOPTIMAL_KHR)==(result)))) ) {
                        {
                                    struct timespec tp ;
            clock_gettime(CLOCK_REALTIME, &tp);
            printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
            printf(".");
            printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" failed to acquire swap chain image.: ");
            printf("\n");
};
};
        VkSemaphore waitSemaphores[]  = {state._imageAvailableSemaphores[state._currentFrame]};
    VkSemaphore signalSemaphores[]  = {state._renderFinishedSemaphores[state._currentFrame]};
    VkPipelineStageFlags waitStages[]  = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    {
                        struct timespec tp ;
        clock_gettime(CLOCK_REALTIME, &tp);
        printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
        printf(".");
        printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" updateUniformBuffer: ");
        printf(" imageIndex=");
        printf(printf_dec_format(imageIndex), imageIndex);
        printf(" (%s)", type_string(imageIndex));
        printf("\n");
};
    updateUniformBuffer(imageIndex);
        VkSubmitInfo submitInfo  = {};
        submitInfo.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount=1;
        submitInfo.pWaitSemaphores=waitSemaphores;
        submitInfo.pWaitDstStageMask=waitStages;
        submitInfo.commandBufferCount=1;
        submitInfo.pCommandBuffers=&(state._commandBuffers[imageIndex]);
        submitInfo.signalSemaphoreCount=1;
        submitInfo.pSignalSemaphores=signalSemaphores;
    vkResetFences(state._device, 1, &(state._inFlightFences[state._currentFrame]));
    if ( !((VK_SUCCESS)==(vkQueueSubmit(state._graphicsQueue, 1, &submitInfo, state._inFlightFences[state._currentFrame]))) ) {
                        {
                                    struct timespec tp ;
            clock_gettime(CLOCK_REALTIME, &tp);
            printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
            printf(".");
            printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" failed to (vkQueueSubmit (dot state _graphicsQueue) 1 &submitInfo            (aref (dot state _inFlightFences) (dot state _currentFrame))): ");
            printf("\n");
};
};
        VkSwapchainKHR swapChains[]  = {state._swapChain};
        VkPresentInfoKHR presentInfo  = {};
        presentInfo.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount=1;
        presentInfo.pWaitSemaphores=signalSemaphores;
        presentInfo.swapchainCount=1;
        presentInfo.pSwapchains=swapChains;
        presentInfo.pImageIndices=&imageIndex;
        presentInfo.pResults=NULL;
    {
                        __auto_type result2  = vkQueuePresentKHR(state._presentQueue, &presentInfo);
        if ( (((VK_SUBOPTIMAL_KHR)==(result2))||((VK_ERROR_OUT_OF_DATE_KHR)==(result2))||(state._framebufferResized)) ) {
                                                state._framebufferResized=false;
            recreateSwapChain();
} else {
                        if ( !((VK_SUCCESS)==(result2)) ) {
                                                {
                                                            struct timespec tp ;
                    clock_gettime(CLOCK_REALTIME, &tp);
                    printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
                    printf(".");
                    printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
                    printf(" ");
                    printf(printf_dec_format(__FILE__), __FILE__);
                    printf(":");
                    printf(printf_dec_format(__LINE__), __LINE__);
                    printf(" ");
                    printf(printf_dec_format(__func__), __func__);
                    printf(" failed to present swap chain image.: ");
                    printf("\n");
};
};
};
};
            state._currentFrame=((1)+(state._currentFrame))%_MAX_FRAMES_IN_FLIGHT;
};