 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void cleanupSwapChain (){
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" cleanupSwapChain: ");
        printf("\n");
};
                vkDestroyImageView(state._device, state._colorImageView, NULL);
    vkDestroyImage(state._device, state._colorImage, NULL);
    vkFreeMemory(state._device, state._colorImageMemory, NULL);
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" cleanup depth: ");
        printf(" state._depthImageView=");
        printf(printf_dec_format(state._depthImageView), state._depthImageView);
        printf(" (%s)", type_string(state._depthImageView));
        printf(" state._depthImage=");
        printf(printf_dec_format(state._depthImage), state._depthImage);
        printf(" (%s)", type_string(state._depthImage));
        printf(" state._depthImageMemory=");
        printf(printf_dec_format(state._depthImageMemory), state._depthImageMemory);
        printf(" (%s)", type_string(state._depthImageMemory));
        printf("\n");
};
    vkDestroyImageView(state._device, state._depthImageView, NULL);
    vkDestroyImage(state._device, state._depthImage, NULL);
    vkFreeMemory(state._device, state._depthImageMemory, NULL);
    for (int b_idx = 0;b_idx<((sizeof(state._swapChainFramebuffers))/(sizeof(*(state._swapChainFramebuffers))));(b_idx)+=(1)) {
                        __auto_type b  = state._swapChainFramebuffers[b_idx];
        {
                        {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" framebuffer: ");
                printf(" b=");
                printf(printf_dec_format(b), b);
                printf(" (%s)", type_string(b));
                printf("\n");
};
                        vkDestroyFramebuffer(state._device, b, NULL);
};
};
    vkFreeCommandBuffers(state._device, state._commandPool, length(state._commandBuffers), state._commandBuffers);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" pipeline: ");
        printf(" state._graphicsPipeline=");
        printf(printf_dec_format(state._graphicsPipeline), state._graphicsPipeline);
        printf(" (%s)", type_string(state._graphicsPipeline));
        printf(" state._pipelineLayout=");
        printf(printf_dec_format(state._pipelineLayout), state._pipelineLayout);
        printf(" (%s)", type_string(state._pipelineLayout));
        printf(" state._renderPass=");
        printf(printf_dec_format(state._renderPass), state._renderPass);
        printf(" (%s)", type_string(state._renderPass));
        printf("\n");
};
    vkDestroyPipeline(state._device, state._graphicsPipeline, NULL);
    vkDestroyPipelineLayout(state._device, state._pipelineLayout, NULL);
    vkDestroyRenderPass(state._device, state._renderPass, NULL);
    for (int view_idx = 0;view_idx<((sizeof(state._swapChainImageViews))/(sizeof(*(state._swapChainImageViews))));(view_idx)+=(1)) {
                        __auto_type view  = state._swapChainImageViews[view_idx];
        {
                        {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" image-view: ");
                printf(" view=");
                printf(printf_dec_format(view), view);
                printf(" (%s)", type_string(view));
                printf("\n");
};
                        vkDestroyImageView(state._device, view, NULL);
};
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
        printf(" swapchain: ");
        printf(" state._swapChain=");
        printf(printf_dec_format(state._swapChain), state._swapChain);
        printf(" (%s)", type_string(state._swapChain));
        printf("\n");
};
    vkDestroySwapchainKHR(state._device, state._swapChain, NULL);
    for (int i = 0;i<length(state._swapChainImages);(i)+=(1)) {
                {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" ubo: ");
            printf(" state._uniformBuffers[i]=");
            printf(printf_dec_format(state._uniformBuffers[i]), state._uniformBuffers[i]);
            printf(" (%s)", type_string(state._uniformBuffers[i]));
            printf(" state._uniformBuffersMemory[i]=");
            printf(printf_dec_format(state._uniformBuffersMemory[i]), state._uniformBuffersMemory[i]);
            printf(" (%s)", type_string(state._uniformBuffersMemory[i]));
            printf("\n");
};
                vkDestroyBuffer(state._device, state._uniformBuffers[i], NULL);
                vkFreeMemory(state._device, state._uniformBuffersMemory[i], NULL);
}
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" descriptor-pool: ");
        printf(" state._descriptorPool=");
        printf(printf_dec_format(state._descriptorPool), state._descriptorPool);
        printf(" (%s)", type_string(state._descriptorPool));
        printf("\n");
};
    vkDestroyDescriptorPool(state._device, state._descriptorPool, NULL);
}
void cleanup (){
            cleanupSwapChain();
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" cleanup: ");
        printf("\n");
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
        printf(" tex: ");
        printf(" state._textureSampler=");
        printf(printf_dec_format(state._textureSampler), state._textureSampler);
        printf(" (%s)", type_string(state._textureSampler));
        printf(" state._textureImageView=");
        printf(printf_dec_format(state._textureImageView), state._textureImageView);
        printf(" (%s)", type_string(state._textureImageView));
        printf(" state._textureImage=");
        printf(printf_dec_format(state._textureImage), state._textureImage);
        printf(" (%s)", type_string(state._textureImage));
        printf(" state._textureImageMemory=");
        printf(printf_dec_format(state._textureImageMemory), state._textureImageMemory);
        printf(" (%s)", type_string(state._textureImageMemory));
        printf(" state._descriptorSetLayout=");
        printf(printf_dec_format(state._descriptorSetLayout), state._descriptorSetLayout);
        printf(" (%s)", type_string(state._descriptorSetLayout));
        printf("\n");
};
    vkDestroySampler(state._device, state._textureSampler, NULL);
    vkDestroyImageView(state._device, state._textureImageView, NULL);
    vkDestroyImage(state._device, state._textureImage, NULL);
    vkFreeMemory(state._device, state._textureImageMemory, NULL);
    vkDestroyDescriptorSetLayout(state._device, state._descriptorSetLayout, NULL);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" buffers: ");
        printf(" state._vertexBuffer=");
        printf(printf_dec_format(state._vertexBuffer), state._vertexBuffer);
        printf(" (%s)", type_string(state._vertexBuffer));
        printf(" state._vertexBufferMemory=");
        printf(printf_dec_format(state._vertexBufferMemory), state._vertexBufferMemory);
        printf(" (%s)", type_string(state._vertexBufferMemory));
        printf(" state._indexBuffer=");
        printf(printf_dec_format(state._indexBuffer), state._indexBuffer);
        printf(" (%s)", type_string(state._indexBuffer));
        printf(" state._indexBufferMemory=");
        printf(printf_dec_format(state._indexBufferMemory), state._indexBufferMemory);
        printf(" (%s)", type_string(state._indexBufferMemory));
        printf("\n");
};
        vkDestroyBuffer(state._device, state._vertexBuffer, NULL);
    vkFreeMemory(state._device, state._vertexBufferMemory, NULL);
        vkDestroyBuffer(state._device, state._indexBuffer, NULL);
    vkFreeMemory(state._device, state._indexBufferMemory, NULL);
    for (int i = 0;i<_MAX_FRAMES_IN_FLIGHT;(i)+=(1)) {
                        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" sync: ");
            printf(" state._renderFinishedSemaphores[i]=");
            printf(printf_dec_format(state._renderFinishedSemaphores[i]), state._renderFinishedSemaphores[i]);
            printf(" (%s)", type_string(state._renderFinishedSemaphores[i]));
            printf(" state._imageAvailableSemaphores[i]=");
            printf(printf_dec_format(state._imageAvailableSemaphores[i]), state._imageAvailableSemaphores[i]);
            printf(" (%s)", type_string(state._imageAvailableSemaphores[i]));
            printf(" state._inFlightFences[i]=");
            printf(printf_dec_format(state._inFlightFences[i]), state._inFlightFences[i]);
            printf(" (%s)", type_string(state._inFlightFences[i]));
            printf("\n");
};
        vkDestroySemaphore(state._device, state._renderFinishedSemaphores[i], NULL);
        vkDestroySemaphore(state._device, state._imageAvailableSemaphores[i], NULL);
        vkDestroyFence(state._device, state._inFlightFences[i], NULL);
}
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" cmd-pool: ");
        printf(" state._commandPool=");
        printf(printf_dec_format(state._commandPool), state._commandPool);
        printf(" (%s)", type_string(state._commandPool));
        printf("\n");
};
    vkDestroyCommandPool(state._device, state._commandPool, NULL);
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" rest: ");
        printf(" state._device=");
        printf(printf_dec_format(state._device), state._device);
        printf(" (%s)", type_string(state._device));
        printf(" state._instance=");
        printf(printf_dec_format(state._instance), state._instance);
        printf(" (%s)", type_string(state._instance));
        printf(" state._window=");
        printf(printf_dec_format(state._window), state._window);
        printf(" (%s)", type_string(state._window));
        printf("\n");
};
        vkDestroyDevice(state._device, NULL);
        vkDestroySurfaceKHR(state._instance, state._surface, NULL);
        vkDestroyInstance(state._instance, NULL);
        glfwDestroyWindow(state._window);
        glfwTerminate();
        cleanupModel();
};