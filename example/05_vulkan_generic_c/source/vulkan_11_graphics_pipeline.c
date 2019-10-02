 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <stdlib.h>
#include <string.h>
 
VkVertexInputBindingDescription Vertex_getBindingDescription (){
                    VkVertexInputBindingDescription bindingDescription  = {};
        bindingDescription.binding=0;
        bindingDescription.stride=sizeof(Vertex);
        bindingDescription.inputRate=VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}
 
VertexInputAttributeDescription3 Vertex_getAttributeDescriptions (){
            VertexInputAttributeDescription3 attributeDescriptions  = {};
        attributeDescriptions.data[0].binding=0;
    attributeDescriptions.data[0].location=0;
    attributeDescriptions.data[0].format=VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions.data[0].offset=offsetof(Vertex, pos);
        attributeDescriptions.data[1].binding=0;
    attributeDescriptions.data[1].location=1;
    attributeDescriptions.data[1].format=VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions.data[1].offset=offsetof(Vertex, color);
        attributeDescriptions.data[2].binding=0;
    attributeDescriptions.data[2].location=2;
    attributeDescriptions.data[2].format=VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions.data[2].offset=offsetof(Vertex, texCoord);
    return attributeDescriptions;
};
 
Array_u8* makeArray_u8 (int n){
            __auto_type n_bytes_Array_u8  = ((sizeof(Array_u8))+(((n)*(sizeof(uint8_t)))));
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
        printf(" malloc: ");
        printf(" n_bytes_Array_u8=");
        printf(printf_dec_format(n_bytes_Array_u8), n_bytes_Array_u8);
        printf(" (%s)", type_string(n_bytes_Array_u8));
        printf("\n");
};
        Array_u8* a  = malloc(n_bytes_Array_u8);
        a->size=n;
    return a;
}
void destroyArray_u8 (Array_u8* a){
        free(a);
}
Array_u8* readFile (const char* filename){
            __auto_type file  = fopen(filename, "r");
    if ( !(file) ) {
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
            printf(" failed to open file.: ");
            printf("\n");
};
};
    if ( file ) {
                        fseek(file, 0L, SEEK_END);
                __auto_type filesize  = ftell(file);
        __auto_type buffer  = makeArray_u8(filesize);
        rewind(file);
                __auto_type read_status  = fread(buffer->data, 1, filesize, file);
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
            printf(" readFile: ");
            printf(" read_status=");
            printf(printf_dec_format(read_status), read_status);
            printf(" (%s)", type_string(read_status));
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf(" filesize=");
            printf(printf_dec_format(filesize), filesize);
            printf(" (%s)", type_string(filesize));
            printf(" file=");
            printf(printf_dec_format(file), file);
            printf(" (%s)", type_string(file));
            printf("\n");
};
        return buffer;
};
}
VkShaderModule createShaderModule (const Array_u8* code){
            VkShaderModule shaderModule ;
    __auto_type codeSize  = code->size;
    __auto_type pCode  = (const uint32_t*)(code->data);
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
        printf(" createShader: ");
        printf(" codeSize=");
        printf(printf_dec_format(codeSize), codeSize);
        printf(" (%s)", type_string(codeSize));
        printf(" pCode=");
        printf(printf_dec_format(pCode), pCode);
        printf(" (%s)", type_string(pCode));
        printf("\n");
};
    {
                        VkShaderModuleCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                info.codeSize=codeSize;
                info.pCode=pCode;
                        if ( !((VK_SUCCESS)==(vkCreateShaderModule(state._device, &info, NULL, &shaderModule))) ) {
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
                printf(" failed to (vkCreateShaderModule (dot state _device) &info NULL &shaderModule): ");
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
            printf("  create shader-module: ");
            printf(" shaderModule=");
            printf(printf_dec_format(shaderModule), shaderModule);
            printf(" (%s)", type_string(shaderModule));
            printf("\n");
};
};
    return shaderModule;
}
void createGraphicsPipeline (){
            __auto_type fv  = readFile("vert.spv");
    __auto_type vertShaderModule  = createShaderModule(fv);
    __auto_type ff  = readFile("frag.spv");
    __auto_type fragShaderModule  = createShaderModule(ff);
    destroyArray_u8(fv);
    destroyArray_u8(ff);
        VkPipelineShaderStageCreateInfo fragShaderStageInfo  = {};
        fragShaderStageInfo.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage=VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module=fragShaderModule;
        fragShaderStageInfo.pName="main";
        fragShaderStageInfo.pSpecializationInfo=NULL;
        VkPipelineShaderStageCreateInfo vertShaderStageInfo  = {};
        vertShaderStageInfo.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage=VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module=vertShaderModule;
        vertShaderStageInfo.pName="main";
        vertShaderStageInfo.pSpecializationInfo=NULL;
        VkPipelineShaderStageCreateInfo shaderStages[]  = {vertShaderStageInfo, fragShaderStageInfo};
    __auto_type bindingDescription  = Vertex_getBindingDescription();
    __auto_type attributeDescriptions  = Vertex_getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo  = {};
        vertexInputInfo.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount=1;
        vertexInputInfo.pVertexBindingDescriptions=&bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount=length(attributeDescriptions.data);
        vertexInputInfo.pVertexAttributeDescriptions=attributeDescriptions.data;
        VkPipelineInputAssemblyStateCreateInfo inputAssembly  = {};
        inputAssembly.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology=VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable=VK_FALSE;
        VkViewport viewport  = {};
        viewport.x=(0.0e+0f);
        viewport.y=(0.0e+0f);
        viewport.width=(((1.e+0f))*(state._swapChainExtent.width));
        viewport.height=(((1.e+0f))*(state._swapChainExtent.height));
        viewport.minDepth=(0.0e+0f);
        viewport.maxDepth=(1.e+0f);
        VkRect2D scissor  = {};
        scissor.offset=(__typeof__(scissor.offset)){0,0};
        scissor.extent=state._swapChainExtent;
        VkPipelineViewportStateCreateInfo viewPortState  = {};
        viewPortState.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewPortState.viewportCount=1;
        viewPortState.pViewports=&viewport;
        viewPortState.scissorCount=1;
        viewPortState.pScissors=&scissor;
        VkPipelineRasterizationStateCreateInfo rasterizer  = {};
        rasterizer.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable=VK_FALSE;
        rasterizer.polygonMode=VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth=(1.e+0f);
        rasterizer.cullMode=VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable=VK_FALSE;
        rasterizer.depthBiasConstantFactor=(0.0e+0f);
        rasterizer.depthBiasClamp=(0.0e+0f);
        rasterizer.depthBiasSlopeFactor=(0.0e+0f);
        VkPipelineMultisampleStateCreateInfo multisampling  = {};
        multisampling.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable=VK_FALSE;
        multisampling.rasterizationSamples=state._msaaSamples;
        multisampling.minSampleShading=(1.e+0f);
        multisampling.pSampleMask=NULL;
        multisampling.alphaToCoverageEnable=VK_FALSE;
        multisampling.alphaToOneEnable=VK_FALSE;
        VkPipelineDepthStencilStateCreateInfo depthStencil  = {};
        depthStencil.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable=VK_TRUE;
        depthStencil.depthWriteEnable=VK_TRUE;
        depthStencil.depthCompareOp=VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable=VK_FALSE;
        depthStencil.minDepthBounds=(0.0e+0f);
        depthStencil.maxDepthBounds=(1.e+0f);
        depthStencil.stencilTestEnable=VK_FALSE;
        depthStencil.front=(__typeof__(depthStencil.front)){};
        depthStencil.back=(__typeof__(depthStencil.back)){};
        VkPipelineColorBlendAttachmentState colorBlendAttachment  = {};
        colorBlendAttachment.colorWriteMask=((VK_COLOR_COMPONENT_R_BIT) | (VK_COLOR_COMPONENT_G_BIT) | (VK_COLOR_COMPONENT_B_BIT) | (VK_COLOR_COMPONENT_A_BIT));
        colorBlendAttachment.blendEnable=VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor=VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor=VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp=VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp=VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo colorBlending  = {};
        colorBlending.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable=VK_FALSE;
        colorBlending.logicOp=VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount=1;
        colorBlending.pAttachments=&colorBlendAttachment;
        colorBlending.blendConstants[0]=(0.0e+0f);
        colorBlending.blendConstants[1]=(0.0e+0f);
        colorBlending.blendConstants[2]=(0.0e+0f);
        colorBlending.blendConstants[3]=(0.0e+0f);
        {
                        VkPipelineLayoutCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                info.setLayoutCount=1;
                info.pSetLayouts=&(state._descriptorSetLayout);
                info.pushConstantRangeCount=0;
                info.pPushConstantRanges=NULL;
                        if ( !((VK_SUCCESS)==(vkCreatePipelineLayout(state._device, &info, NULL, &(state._pipelineLayout)))) ) {
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
                printf(" failed to (vkCreatePipelineLayout (dot state _device) &info NULL            (ref (dot state _pipelineLayout))): ");
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
            printf("  create pipeline-layout: ");
            printf(" state._pipelineLayout=");
            printf(printf_dec_format(state._pipelineLayout), state._pipelineLayout);
            printf(" (%s)", type_string(state._pipelineLayout));
            printf("\n");
};
};
    {
                        VkGraphicsPipelineCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                info.stageCount=2;
                info.pStages=shaderStages;
                info.pVertexInputState=&vertexInputInfo;
                info.pInputAssemblyState=&inputAssembly;
                info.pViewportState=&viewPortState;
                info.pRasterizationState=&rasterizer;
                info.pMultisampleState=&multisampling;
                info.pDepthStencilState=&depthStencil;
                info.pColorBlendState=&colorBlending;
                info.pDynamicState=NULL;
                info.layout=state._pipelineLayout;
                info.renderPass=state._renderPass;
                info.subpass=0;
                info.basePipelineHandle=VK_NULL_HANDLE;
                info.basePipelineIndex=-1;
                        if ( !((VK_SUCCESS)==(vkCreateGraphicsPipelines(state._device, VK_NULL_HANDLE, 1, &info, NULL, &(state._graphicsPipeline)))) ) {
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
                printf(" failed to (vkCreateGraphicsPipelines (dot state _device) VK_NULL_HANDLE 1 &info            NULL (ref (dot state _graphicsPipeline))): ");
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
            printf("  create graphics-pipeline: ");
            printf(" state._graphicsPipeline=");
            printf(printf_dec_format(state._graphicsPipeline), state._graphicsPipeline);
            printf(" (%s)", type_string(state._graphicsPipeline));
            printf("\n");
};
};
    vkDestroyShaderModule(state._device, fragShaderModule, NULL);
    vkDestroyShaderModule(state._device, vertShaderModule, NULL);
};