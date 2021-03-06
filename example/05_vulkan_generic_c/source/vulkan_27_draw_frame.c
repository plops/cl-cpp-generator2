 
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
    return (((double) tp.tv_sec)+((((1.e-9))*(tp.tv_nsec))));
}
void updateUniformBuffer (uint32_t currentImage){
            static double startTime ;
    if ( ((0.0e+0))==(startTime) ) {
                                startTime=now();
};
        double currentTime  = now();
    double time  = ((currentTime)-(startTime));
        __auto_type zAxis  = (vec3) {(0.0e+0f), (0.0e+0f), (1.e+0f)};
    __auto_type eye  = (vec3) {(2.e+0f), (2.e+0f), (2.e+0f)};
    __auto_type center  = (vec3) {(0.0e+0f), (0.0e+0f), (0.0e+0f)};
    __auto_type angularRate  = glm_rad((9.e+0f));
    __auto_type rotationAngle  = (float) ((time)*(angularRate));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
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
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
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
            vkWaitForFences(state._device, 1, &(state._inFlightFences[state._currentFrame]), VK_TRUE, UINT64_MAX);
            uint32_t imageIndex  = 0;
    __auto_type result  = vkAcquireNextImageKHR(state._device, state._swapChain, UINT64_MAX, state._imageAvailableSemaphores[state._currentFrame], VK_NULL_HANDLE, &imageIndex);
    if ( (VK_ERROR_OUT_OF_DATE_KHR)==(result) ) {
                        recreateSwapChain();
        return ;
};
    if ( !((((VK_SUCCESS)==(result))||((VK_SUBOPTIMAL_KHR)==(result)))) ) {
                        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                                            __auto_type current_time  = now();
                    printf("%6.6f", ((current_time)-(state._start_time)));
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