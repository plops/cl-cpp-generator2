#ifndef UTILS_H
 
#define UTILS_H
 
#include <stdio.h>
 
#include <stdbool.h>
 
#include <time.h>
 
#include <cglm/cglm.h>
 
struct QueueFamilyIndices {
        int graphicsFamily;
        int presentFamily;
};
typedef struct QueueFamilyIndices QueueFamilyIndices;
struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        int formatsCount;
        VkSurfaceFormatKHR* formats;
        int presentModesCount;
        VkPresentModeKHR* presentModes;
};
typedef struct SwapChainSupportDetails SwapChainSupportDetails;
struct Vertex {
        vec3 pos;
        vec3 color;
        vec2 texCoord;
};
typedef struct Vertex Vertex;
struct VertexInputAttributeDescription3 {
        VkVertexInputAttributeDescription data[3];
};
typedef struct VertexInputAttributeDescription3 VertexInputAttributeDescription3;
struct Array_u8 {
        int size;
        uint8_t* data[];
};
typedef struct Array_u8 Array_u8;
struct Tuple_Image_DeviceMemory {
        VkImage image;
        VkDeviceMemory memory;
};
typedef struct Tuple_Image_DeviceMemory Tuple_Image_DeviceMemory;
struct Triple_FrambufferViews {
        VkImageView image;
        VkImageView depth;
        VkImageView swap;
};
typedef struct Triple_FrambufferViews Triple_FrambufferViews;
struct Tuple_Buffer_DeviceMemory {
        VkBuffer buffer;
        VkDeviceMemory memory;
};
typedef struct Tuple_Buffer_DeviceMemory Tuple_Buffer_DeviceMemory;
struct mmapPair {
        int n;
        char* data;
};
typedef struct mmapPair mmapPair;
struct Hashmap_int_data {
        int value;
        int count;
        uint64_t hash;
};
typedef struct Hashmap_int_data Hashmap_int_data;
struct Hashmap_int {
        int n;
        int bins;
        int n_entries;
        Hashmap_int_data* data;
};
typedef struct Hashmap_int Hashmap_int;
struct Hashmap_int_pair {
        int key;
        Hashmap_int_data value;
        Hashmap_int_data* valuep;
};
typedef struct Hashmap_int_pair Hashmap_int_pair;
struct Geometry_store_head {
        uint64_t header;
        uint64_t version;
        uint64_t date;
        int _num_vertices;
        int _num_indices;
};
typedef struct Geometry_store_head Geometry_store_head;
struct Geometry_store {
        uint64_t header;
        uint64_t version;
        uint64_t date;
        int _num_vertices;
        int _num_indices;
        Vertex _vertices[265645];
        uint32_t _indices[1500000];
};
typedef struct Geometry_store Geometry_store;
struct UniformBufferObject {
        mat4 model;
        mat4 view;
        mat4 proj;
};
typedef struct UniformBufferObject UniformBufferObject;
#define length(a) (sizeof((a))/sizeof(*(a)))
#define max(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a < _b ? _a : _b; })
#define printf_dec_format(x) _Generic((x), default: "%p", char: "%c", signed char: "%hhd", unsigned char: "%hhu", signed short: "%hd", unsigned short: "%hu", signed int: "%d", unsigned int: "%u", long int: "%ld", unsigned long int: "%lu", long long int: "%lld", float: "%f", double: "%f", long double: "%Lf", char*: "%s", const char*: "%s", unsigned long long int: "%llu",void*: "%p",bool:"%d")
#define type_string(x) _Generic((x), default: "default",bool: "bool",const bool: "const bool",char: "char",const char: "const char",unsigned char: "unsigned char",const unsigned char: "const unsigned char",short: "short",const short: "const short",unsigned short: "unsigned short",const unsigned short: "const unsigned short",int: "int",const int: "const int",unsigned int: "unsigned int",const unsigned int: "const unsigned int",long int: "long int",const long int: "const long int",unsigned long int: "unsigned long int",const unsigned long int: "const unsigned long int",long long int: "long long int",const long long int: "const long long int",unsigned long long int: "unsigned long long int",const unsigned long long int: "const unsigned long long int",float: "float",const float: "const float",double: "double",const double: "const double",long double: "long double",const long double: "const long double",char*: "char*",const char*: "const char*",void*: "void*",const void*: "const void*")
 ;
 
#endif
 