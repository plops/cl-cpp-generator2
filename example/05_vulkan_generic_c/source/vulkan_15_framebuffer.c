 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
 
void createFramebuffers (){
            __auto_type n  = length(state._swapChainImageViews);
    for (int i = 0;i<n;(i)+=(1)) {
                        __auto_type attachments  = (Triple_FrambufferViews) {state._colorImageView, state._depthImageView, state._swapChainImageViews[i]};
        {
                                    VkFramebufferCreateInfo info  = {};
                        info.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                        info.renderPass=state._renderPass;
                        info.attachmentCount=3;
                        info.pAttachments=(VkImageView*) &attachments;
                        info.width=state._swapChainExtent.width;
                        info.height=state._swapChainExtent.height;
                        info.layers=1;
                                    if ( !((VK_SUCCESS)==(vkCreateFramebuffer(state._device, &info, NULL, &(state._swapChainFramebuffers[i])))) ) {
                                                {
                                                            __auto_type current_time  = now();
                    printf("%6.6f", ((current_time)-(state._start_time)));
                    printf(" ");
                    printf(printf_dec_format(__FILE__), __FILE__);
                    printf(":");
                    printf(printf_dec_format(__LINE__), __LINE__);
                    printf(" ");
                    printf(printf_dec_format(__func__), __func__);
                    printf(" failed to (vkCreateFramebuffer (dot state _device) &info NULL            (ref (aref (dot state _swapChainFramebuffers) i))): ");
                    printf("\n");
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
                printf("  create framebuffer: ");
                printf(" state._swapChainFramebuffers[i]=");
                printf(printf_dec_format(state._swapChainFramebuffers[i]), state._swapChainFramebuffers[i]);
                printf(" (%s)", type_string(state._swapChainFramebuffers[i]));
                printf("\n");
};
};
};
};