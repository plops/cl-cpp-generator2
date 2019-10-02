 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
 
void munmapFile (mmapPair pair){
        munmap(pair.data, pair.n);
}
mmapPair mmapFile (char* filename){
            __auto_type f  = fopen(filename, "r");
    if ( !(f) ) {
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
            printf(" can't open file: ");
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf("\n");
};
};
    fseek(f, 0, SEEK_END);
        __auto_type filesize  = ftell(f);
    fclose(f);
            __auto_type fd  = open(filename, O_RDONLY);
    struct stat sb ;
    if ( (-1)==(fd) ) {
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
            printf(" can't open file for mmap: ");
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf("\n");
};
};
    if ( (-1)==(fstat(fd, &sb)) ) {
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
            printf(" can't fstat file: ");
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf("\n");
};
};
    if ( !(S_ISREG(sb.st_mode)) ) {
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
            printf(" not a file: ");
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf(" sb.st_mode=");
            printf(printf_dec_format(sb.st_mode), sb.st_mode);
            printf(" (%s)", type_string(sb.st_mode));
            printf("\n");
};
};
        __auto_type p  = mmap(0, filesize, PROT_READ, MAP_SHARED, fd, 0);
    if ( (MAP_FAILED)==(p) ) {
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
            printf(" mmap failed: ");
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf("\n");
};
}
    if ( (-1)==(close(fd)) ) {
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
            printf(" close failed: ");
            printf(" filename=");
            printf(printf_dec_format(filename), filename);
            printf(" (%s)", type_string(filename));
            printf("\n");
};
}
        __auto_type map  = (mmapPair) {filesize, p};
    return map;
}
void loadModel (){
            __auto_type map  = mmapFile("chalet.obj");
    tinyobj_attrib_t attrib ;
    tinyobj_shape_t* shapes  = NULL;
    size_t num_shapes ;
    tinyobj_material_t* materials  = NULL;
    size_t num_materials ;
    __auto_type res  = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, &materials, &num_materials, map.data, map.n, TINYOBJ_FLAG_TRIANGULATE);
    if ( !((TINYOBJ_SUCCESS)==(res)) ) {
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
            printf(" tinyobj failed to open: ");
            printf(" res=");
            printf(printf_dec_format(res), res);
            printf(" (%s)", type_string(res));
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
        printf(" model: ");
        printf(" num_shapes=");
        printf(printf_dec_format(num_shapes), num_shapes);
        printf(" (%s)", type_string(num_shapes));
        printf(" num_materials=");
        printf(printf_dec_format(num_materials), num_materials);
        printf(" (%s)", type_string(num_materials));
        printf(" attrib.num_face_num_verts=");
        printf(printf_dec_format(attrib.num_face_num_verts), attrib.num_face_num_verts);
        printf(" (%s)", type_string(attrib.num_face_num_verts));
        printf(" attrib.num_vertices=");
        printf(printf_dec_format(attrib.num_vertices), attrib.num_vertices);
        printf(" (%s)", type_string(attrib.num_vertices));
        printf(" attrib.num_texcoords=");
        printf(printf_dec_format(attrib.num_texcoords), attrib.num_texcoords);
        printf(" (%s)", type_string(attrib.num_texcoords));
        printf(" attrib.num_faces=");
        printf(printf_dec_format(attrib.num_faces), attrib.num_faces);
        printf(" (%s)", type_string(attrib.num_faces));
        printf(" attrib.num_normals=");
        printf(printf_dec_format(attrib.num_normals), attrib.num_normals);
        printf(" (%s)", type_string(attrib.num_normals));
        printf("\n");
};
        state._num_vertices=(int) ((attrib.num_faces)/(3));
    state._vertices=malloc(((sizeof(*(state._vertices)))*(state._num_vertices)));
    state._num_indices=(int) ((attrib.num_faces)/(3));
    state._indices=malloc(((sizeof(*(state._indices)))*(state._num_indices)));
    for (int j = 0;j<state._num_vertices;(j)+=(1)) {
                        __auto_type vertex0  = attrib.vertices[((0)+(((3)*(j))))];
        __auto_type vertex1  = attrib.vertices[((1)+(((3)*(j))))];
        __auto_type vertex2  = attrib.vertices[((2)+(((3)*(j))))];
        __auto_type vertex  = (Vertex) {{vertex0, vertex1, vertex2}, {(1.e+0f), (1.e+0f), (1.e+0f)}, {(0.0e+0f), (0.0e+0f)}};
                state._vertices[j]=vertex;
        state._indices[j]=j;
}
    munmapFile(map);
};