#ifndef PROTO2_H
#define PROTO2_H
 void mainLoop ();
 void run ();
 void cleanupInstance ();
 void createInstance ();
 void initVulkan ();
 void initWindow ();
 void cleanupWindow ();
 void cleanupSurface ();
 void createSurface ();
 void cleanupPhysicalDevice ();
 _Bool QueueFamilyIndices_isComplete (QueueFamilyIndices q);
 QueueFamilyIndices QueueFamilyIndices_make ();
 void QueueFamilyIndices_destroy (QueueFamilyIndices* q);
 QueueFamilyIndices findQueueFamilies (VkPhysicalDevice device);
 void cleanupSwapChainSupport (SwapChainSupportDetails* details);
 SwapChainSupportDetails querySwapChainSupport (VkPhysicalDevice device);
 bool isDeviceSuitable (VkPhysicalDevice device);
 bool checkDeviceExtensionSupport (VkPhysicalDevice device);
 VkSampleCountFlagBits getMaxUsableSampleCount ();
 void pickPhysicalDevice ();
 void cleanupLogicalDevice ();
 void createLogicalDevice ();
 VkSurfaceFormatKHR chooseSwapSurfaceFormat (const VkSurfaceFormatKHR* availableFormats, int n);
 VkPresentModeKHR chooseSwapPresentMode (const VkPresentModeKHR* modes, int n);
 VkExtent2D chooseSwapExtent (const VkSurfaceCapabilitiesKHR* capabilities);
 void createSwapChain ();
 void cleanupImageView ();
 VkImageView createImageView (VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
 void createImageViews ();
 void cleanupRenderPass ();
 VkFormat findSupportedFormat (VkFormat* candidates, int n, VkImageTiling tiling, VkFormatFeatureFlags features);
 VkFormat findDepthFormat ();
 void createRenderPass ();
 void createDescriptorSetLayout ();
 VkVertexInputBindingDescription Vertex_getBindingDescription ();
 VertexInputAttributeDescription3 Vertex_getAttributeDescriptions ();
 Array_u8* makeArray_u8 (int n);
 void destroyArray_u8 (Array_u8* a);
 Array_u8* readFile (const char* filename);
 VkShaderModule createShaderModule (const Array_u8* code);
 void createGraphicsPipeline ();
 void createCommandPool ();
 bool hasStencilComponent (VkFormat format);
 VkCommandBuffer beginSingleTimeCommands ();
 void endSingleTimeCommands (VkCommandBuffer commandBuffer);
 void transitionImageLayout (VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
 uint32_t findMemoryType (uint32_t typeFilter, VkMemoryPropertyFlags properties);
 Tuple_Image_DeviceMemory makeTuple_Image_DeviceMemory (VkImage image, VkDeviceMemory memory);
 Tuple_Image_DeviceMemory createImage (uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties);
 void createColorResources ();
 void createDepthResources ();
 void createFramebuffers ();
 Tuple_Buffer_DeviceMemory createBuffer (VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
 void generateMipmaps (VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, int32_t mipLevels);
 void copyBufferToImage (VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
 void createTextureImage ();
 void createTextureImageView ();
 void createTextureSampler ();
 char *dynamic_fgets(char **buf, size_t *size, FILE *file) ;
 void munmapFile (mmapPair pair);
 mmapPair mmapFile (char* filename);
 void cleanupModel ();
 uint64_t hash_i64 (uint64_t u);
 uint64_t hash_combine (uint64_t seed, uint64_t hash);
 uint64_t hash_array_f32 (float* a, int n);
 uint64_t unaligned_load (const char* p);
 uint64_t load_bytes (const char* p, int n);
 uint64_t shift_mix (uint64_t v);
 uint64_t hash_bytes (const void* ptr, uint64_t len);
 uint64_t hash_f32 (float f);
 uint64_t hash_Vertex (Vertex* v);
 Hashmap_int hashmap_int_make (int n, int bins);
 void hashmap_int_free (Hashmap_int* h);
 Hashmap_int_pair hashmap_int_get (Hashmap_int* h, uint64_t key, int bin);
 Hashmap_int_pair hashmap_int_search (Hashmap_int* h, uint64_t key);
 bool hashmap_int_set (Hashmap_int* h, uint64_t key, int newvalue);
 bool equalp_Vertex (Vertex* a, Vertex* b);
 int next_power_of_two (int n);
 void saveCachedModel ();
 bool loadCachedModel ();
 void loadModel ();
 void loadModel_from_obj ();
 void copyBuffer (VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
 void createVertexBuffer ();
 void createIndexBuffer ();
 void createUniformBuffers ();
 void createDescriptorPool ();
 void createDescriptorSets ();
 void createCommandBuffers ();
 void createSyncObjects ();
 double now ();
 void updateUniformBuffer (uint32_t currentImage);
 void recreateSwapChain ();
 void drawFrame ();
 void cleanupSwapChain ();
 void cleanup ();
#endif
