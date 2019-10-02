 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#pragma GCC optimize ("O3")
 
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
void cleanupModel (){
        free(state._vertices);
        free(state._indices);
            state._num_vertices=0;
    state._num_indices=0;
}
void loadModel (){
            __auto_type map  = mmapFile("chalet.obj");
    tinyobj_attrib_t attrib ;
    tinyobj_shape_t* shapes  = NULL;
    size_t num_shapes ;
    tinyobj_material_t* materials  = NULL;
    size_t num_materials ;
    tinyobj_attrib_init(&attrib);
        __auto_type res  = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, &materials, &num_materials, map.data, map.n, TINYOBJ_FLAG_TRIANGULATE);
    if ( !((TINYOBJ_SUCCESS)==(res)) ) {
                        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
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
        __auto_type n_bytes_vertices  = ((sizeof(*(state._vertices)))*(state._num_vertices));
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" malloc: ");
        printf(" n_bytes_vertices=");
        printf(printf_dec_format(n_bytes_vertices), n_bytes_vertices);
        printf(" (%s)", type_string(n_bytes_vertices));
        printf("\n");
};
        state._vertices=malloc(n_bytes_vertices);
            state._num_indices=(int) ((attrib.num_faces)/(3));
        __auto_type n_bytes_indices  = ((sizeof(*(state._indices)))*(state._num_indices));
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" malloc: ");
        printf(" n_bytes_indices=");
        printf(printf_dec_format(n_bytes_indices), n_bytes_indices);
        printf(" (%s)", type_string(n_bytes_indices));
        printf("\n");
};
        state._indices=malloc(n_bytes_indices);
    for (int i = 0;i<500000;(i)+=(1)) {
                        __auto_type v0  = attrib.vertices[((0)+(((3)*(attrib.faces[i].v_idx))))];
        __auto_type v1  = attrib.vertices[((1)+(((3)*(attrib.faces[i].v_idx))))];
        __auto_type v2  = attrib.vertices[((2)+(((3)*(attrib.faces[i].v_idx))))];
        __auto_type t0  = attrib.texcoords[((0)+(((2)*(attrib.faces[i].v_idx))))];
        __auto_type t1  = attrib.texcoords[((1)+(((2)*(attrib.faces[i].v_idx))))];
        __auto_type vertex  = (Vertex) {{v0, v1, v2}, {(1.e+0f), (1.e+0f), (1.e+0f)}, {t0, t1}};
                state._vertices[i]=vertex;
        state._indices[i]=i;
}
    munmapFile(map);
        // cleanup
    tinyobj_attrib_free(&attrib);
    if ( shapes ) {
                        tinyobj_shapes_free(shapes, num_shapes);
};
    if ( materials ) {
                        tinyobj_materials_free(materials, num_materials);
};
};