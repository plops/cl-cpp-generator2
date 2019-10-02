 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void cleanupImageView (){
}
VkImageView createImageView (VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels){
            VkImageView imageView ;
    {
                        VkImageViewCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                info.image=image;
                info.viewType=VK_IMAGE_VIEW_TYPE_2D;
                info.format=format;
                info.subresourceRange.aspectMask=aspectFlags;
                info.subresourceRange.baseMipLevel=0;
                info.subresourceRange.levelCount=mipLevels;
                info.subresourceRange.baseArrayLayer=0;
                info.subresourceRange.layerCount=1;
                        if ( !((VK_SUCCESS)==(vkCreateImageView(state._device, &info, NULL, &imageView))) ) {
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
                printf(" failed to (vkCreateImageView (dot state _device) &info NULL &imageView): ");
                printf("\n");
};
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
            printf("  create image-view: ");
            printf(" imageView=");
            printf(printf_dec_format(imageView), imageView);
            printf(" (%s)", type_string(imageView));
            printf("\n");
};
};
    return imageView;
}
void createImageViews (){
        for (int i = 0;i<length(state._swapChainImages);(i)+=(1)) {
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
            printf(" createImageView: ");
            printf(" i=");
            printf(printf_dec_format(i), i);
            printf(" (%s)", type_string(i));
            printf(" length(state._swapChainImages)=");
            printf(printf_dec_format(length(state._swapChainImages)), length(state._swapChainImages));
            printf(" (%s)", type_string(length(state._swapChainImages)));
            printf("\n");
};
                        state._swapChainImageViews[i]=createImageView(state._swapChainImages[i], state._swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}
};