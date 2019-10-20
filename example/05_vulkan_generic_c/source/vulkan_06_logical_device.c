 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void cleanupLogicalDevice (){
        vkDestroyDevice(state._device, NULL);
}
void createLogicalDevice (){
        // initialize members _device and _graphicsQueue
            __auto_type indices  = findQueueFamilies(state._physicalDevice);
    float queuePriority  = (1.e+0f);
        int allQueueFamilies[]  = {indices.graphicsFamily, indices.presentFamily};
    const int qNumber  = length(allQueueFamilies);
    uint32_t qSeen[qNumber] ;
    __auto_type qSeenCount  = 0;
    for (int q_idx = 0;q_idx<((sizeof(allQueueFamilies))/(sizeof(*(allQueueFamilies))));(q_idx)+=(1)) {
                        __auto_type q  = allQueueFamilies[q_idx];
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
                printf(" check if queue is valid and was seen before: ");
                printf(" q=");
                printf(printf_dec_format(q), q);
                printf(" (%s)", type_string(q));
                printf("\n");
};
                        if ( !((-1)==(q)) ) {
                                                                __auto_type n  = qSeenCount;
                if ( (n)==(0) ) {
                                                            {
                                                                        __auto_type current_time  = now();
                        printf("%6.6f", ((current_time)-(state._start_time)));
                        printf(" ");
                        printf(printf_dec_format(__FILE__), __FILE__);
                        printf(":");
                        printf(printf_dec_format(__LINE__), __LINE__);
                        printf(" ");
                        printf(printf_dec_format(__func__), __func__);
                        printf(" first entry: ");
                        printf(" n=");
                        printf(printf_dec_format(n), n);
                        printf(" (%s)", type_string(n));
                        printf("\n");
};
                                        qSeen[0]=q;
                    (qSeenCount)++;
} else {
                                                            for (int i = 0;i<n;(i)+=(1)) {
                                                {
                                                                                    __auto_type current_time  = now();
                            printf("%6.6f", ((current_time)-(state._start_time)));
                            printf(" ");
                            printf(printf_dec_format(__FILE__), __FILE__);
                            printf(":");
                            printf(printf_dec_format(__LINE__), __LINE__);
                            printf(" ");
                            printf(printf_dec_format(__func__), __func__);
                            printf(" loop through all queue indeces that have been seen before: ");
                            printf(" qSeen[i]=");
                            printf(printf_dec_format(qSeen[i]), qSeen[i]);
                            printf(" (%s)", type_string(qSeen[i]));
                            printf(" i=");
                            printf(printf_dec_format(i), i);
                            printf(" (%s)", type_string(i));
                            printf(" n=");
                            printf(printf_dec_format(n), n);
                            printf(" (%s)", type_string(n));
                            printf("\n");
};
                                                if ( (q)==(qSeen[i]) ) {
                                                                                    {
                                                                                                __auto_type current_time  = now();
                                printf("%6.6f", ((current_time)-(state._start_time)));
                                printf(" ");
                                printf(printf_dec_format(__FILE__), __FILE__);
                                printf(":");
                                printf(printf_dec_format(__LINE__), __LINE__);
                                printf(" ");
                                printf(printf_dec_format(__func__), __func__);
                                printf(" seen before: ");
                                printf(" q=");
                                printf(printf_dec_format(q), q);
                                printf(" (%s)", type_string(q));
                                printf(" qSeenCount=");
                                printf(printf_dec_format(qSeenCount), qSeenCount);
                                printf(" (%s)", type_string(qSeenCount));
                                printf(" n=");
                                printf(printf_dec_format(n), n);
                                printf(" (%s)", type_string(n));
                                printf("\n");
};
                            break;
} else {
                                                                                    {
                                                                                                __auto_type current_time  = now();
                                printf("%6.6f", ((current_time)-(state._start_time)));
                                printf(" ");
                                printf(printf_dec_format(__FILE__), __FILE__);
                                printf(":");
                                printf(printf_dec_format(__LINE__), __LINE__);
                                printf(" ");
                                printf(printf_dec_format(__func__), __func__);
                                printf(" not seen before: ");
                                printf(" q=");
                                printf(printf_dec_format(q), q);
                                printf(" (%s)", type_string(q));
                                printf(" qSeenCount=");
                                printf(printf_dec_format(qSeenCount), qSeenCount);
                                printf(" (%s)", type_string(qSeenCount));
                                printf(" n=");
                                printf(printf_dec_format(n), n);
                                printf(" (%s)", type_string(n));
                                printf("\n");
};
                                                        qSeen[((i)+(1))]=q;
                            (qSeenCount)++;
}
};
};
};
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
        printf(" seen: ");
        printf(" qSeenCount=");
        printf(printf_dec_format(qSeenCount), qSeenCount);
        printf(" (%s)", type_string(qSeenCount));
        printf("\n");
};
        VkDeviceQueueCreateInfo queueCreateInfos[qSeenCount] ;
    int uniqueQueueFamilies[qSeenCount] ;
    __auto_type info_count  = 0;
    for (int i = 0;i<qSeenCount;(i)+=(1)) {
                        __auto_type q  = qSeen[i];
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" copy qSeen into uniquQueueFamilies: ");
            printf(" i=");
            printf(printf_dec_format(i), i);
            printf(" (%s)", type_string(i));
            printf(" q=");
            printf(printf_dec_format(q), q);
            printf(" (%s)", type_string(q));
            printf(" qSeenCount=");
            printf(printf_dec_format(qSeenCount), qSeenCount);
            printf(" (%s)", type_string(qSeenCount));
            printf("\n");
};
                uniqueQueueFamilies[i]=q;
}
    for (int queueFamily_idx = 0;queueFamily_idx<((sizeof(uniqueQueueFamilies))/(sizeof(*(uniqueQueueFamilies))));(queueFamily_idx)+=(1)) {
                        __auto_type queueFamily  = uniqueQueueFamilies[queueFamily_idx];
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
                printf(" create unique queue: ");
                printf(" queueFamily=");
                printf(printf_dec_format(queueFamily), queueFamily);
                printf(" (%s)", type_string(queueFamily));
                printf(" info_count=");
                printf(printf_dec_format(info_count), info_count);
                printf(" (%s)", type_string(info_count));
                printf("\n");
};
                                    VkDeviceQueueCreateInfo queueCreateInfo  = {};
                        queueCreateInfo.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                        queueCreateInfo.queueFamilyIndex=queueFamily;
                        queueCreateInfo.queueCount=1;
                        queueCreateInfo.pQueuePriorities=&queuePriority;
                                    queueCreateInfos[info_count]=queueCreateInfo;
                        (info_count)++;
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
                    printf(" created unique queue: ");
                    printf(" queueFamily=");
                    printf(printf_dec_format(queueFamily), queueFamily);
                    printf(" (%s)", type_string(queueFamily));
                    printf(" info_count=");
                    printf(printf_dec_format(info_count), info_count);
                    printf(" (%s)", type_string(info_count));
                    printf(" queueCreateInfo=");
                    printf(printf_dec_format(queueCreateInfo), queueCreateInfo);
                    printf(" (%s)", type_string(queueCreateInfo));
                    printf("\n");
};
};
};
};
        VkPhysicalDeviceFeatures deviceFeatures  = {};
        deviceFeatures.samplerAnisotropy=VK_TRUE;
    {
                        VkDeviceCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                info.pQueueCreateInfos=queueCreateInfos;
                info.queueCreateInfoCount=length(queueCreateInfos);
                info.pEnabledFeatures=&deviceFeatures;
                info.enabledExtensionCount=length(state._deviceExtensions);
                info.ppEnabledExtensionNames=state._deviceExtensions;
                info.enabledLayerCount=length(state._validationLayers);
                info.ppEnabledLayerNames=state._validationLayers;
                        if ( !((VK_SUCCESS)==(vkCreateDevice(state._physicalDevice, &info, NULL, &(state._device)))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkCreateDevice (dot state _physicalDevice) &info NULL            (ref (dot state _device))): ");
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
            printf("  create device: ");
            printf(" state._physicalDevice=");
            printf(printf_dec_format(state._physicalDevice), state._physicalDevice);
            printf(" (%s)", type_string(state._physicalDevice));
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
        printf(" after create device : ");
        printf(" state._validationLayers=");
        printf(printf_dec_format(state._validationLayers), state._validationLayers);
        printf(" (%s)", type_string(state._validationLayers));
        printf(" length(state._validationLayers)=");
        printf(printf_dec_format(length(state._validationLayers)), length(state._validationLayers));
        printf(" (%s)", type_string(length(state._validationLayers)));
        printf("\n");
};
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
            printf(" create graphics queue: ");
            printf(" indices.graphicsFamily=");
            printf(printf_dec_format(indices.graphicsFamily), indices.graphicsFamily);
            printf(" (%s)", type_string(indices.graphicsFamily));
            printf("\n");
};
                vkGetDeviceQueue(state._device, indices.graphicsFamily, 0, &(state._graphicsQueue));
};
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
            printf(" create present queue: ");
            printf(" indices.presentFamily=");
            printf(printf_dec_format(indices.presentFamily), indices.presentFamily);
            printf(" (%s)", type_string(indices.presentFamily));
            printf("\n");
};
                vkGetDeviceQueue(state._device, indices.presentFamily, 0, &(state._presentQueue));
};
};