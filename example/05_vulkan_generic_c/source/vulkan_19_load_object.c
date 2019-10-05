 
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
uint64_t hash_i64 (uint64_t u){
            uint64_t v  = ((((u)*(3935559000370003845UL)))+(2691343689449507681UL));
    (v)^=((v)>>(21));
    (v)^=((v)<<(37));
    (v)^=((v)>>(4));
    v*=(4768777513237032717UL);
    (v)^=((v)<<(20));
    (v)^=((v)>>(41));
    (v)^=((v)<<(5));
    return v;
}
uint64_t hash_combine (uint64_t seed, uint64_t hash){
        // from glm/gtx/hash.inl
        (hash)+=(((0x9e3779b9)+((seed)<<(6))+((seed)>>(2))));
        return ((seed)^(hash));
}
uint64_t hash_array_f32 (float* a, int n){
            uint64_t seed  = 0;
    for (int i = 0;i<n;(i)+=(1)) {
                        seed=hash_combine(seed, hash_f32(a[i]));
}
    return seed;
}
uint64_t unaligned_load (const char* p){
            uint64_t result ;
    __builtin_memcpy(&result, p, sizeof(result));
    return result;
}
uint64_t load_bytes (const char* p, int n){
        // 1<=n<8
            uint64_t result  = 0;
    for (int i=((n)-(1));(0)<=(i);(i)--) {
                        result=(((result)<<(8))+((unsigned char) p[i]));
}
    return result;
}
uint64_t shift_mix (uint64_t v){
        return ((v)^((v)>>(47)));
}
uint64_t hash_bytes (const void* ptr, uint64_t len){
        // /usr/include/c++/9.1.0/bits/hash_bytes.h 
        // https://github.com/Alexpux/GCC/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/hash_bytes.cc
        // Murmur hash for 64-bit size_t
        // https://stackoverflow.com/questions/11899616/murmurhash-what-is-it
            const uint64_t seed  = 0xc70f6907UL;
    const uint64_t mul  = (((0xc6a4a793UL)<<(32UL))+(0x5bd1e995UL));
    __auto_type buf  = (const void*) ptr;
    const int len_aligned  = ((len)&(~0x7));
    __auto_type end  = ((buf)+(len_aligned));
    uint64_t hash  = ((seed)^(((len)*(mul))));
    for (const char* p=buf;(p)!=(end);(p)+=(8)) {
                        const uint64_t data  = ((shift_mix(((unaligned_load(p))*(mul))))*(mul));
        (hash)^=(data);
        hash*=(mul);
}
    if ( !((0)==(((len)&(0x7)))) ) {
                                const uint64_t data  = load_bytes(end, ((len)&(0x7)));
        (hash)^=(data);
        hash*=(mul);
};
        hash=((shift_mix(hash))*(mul));
    hash=shift_mix(hash);
    return hash;
}
uint64_t hash_f32 (float f){
        // convert float to 64 bit double and consider the double as a 64bit uint to compute hash
        // /usr/include/c++/9.1.0/bits/functional_hash.h
        if ( (f)==((0.0e+0f)) ) {
                        return 0;
};
        return hash_bytes(&f, sizeof(f));
}
uint64_t hash_Vertex (Vertex* v){
            __auto_type pos  = hash_array_f32(v->pos, length(v->pos));
    __auto_type col  = hash_array_f32(v->color, length(v->color));
    __auto_type tex  = hash_array_f32(v->texCoord, length(v->texCoord));
    // http://en.cppreference.com/w/cpp/utility/hash Discussion
    // consecutive identical hashes can delete each other
    return ((pos)^(((col)<<(1))>>(1))^((tex)<<(1)));
}
 
Hashmap_int hashmap_int_make (int n, int bins){
        // initialize hash map with -1
            __auto_type n_bytes_hashmap  = ((sizeof(Hashmap_int_data))*(n)*(bins));
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
        printf(" n_bytes_hashmap=");
        printf(printf_dec_format(n_bytes_hashmap), n_bytes_hashmap);
        printf(" (%s)", type_string(n_bytes_hashmap));
        printf(" n=");
        printf(printf_dec_format(n), n);
        printf(" (%s)", type_string(n));
        printf(" bins=");
        printf(printf_dec_format(bins), bins);
        printf(" (%s)", type_string(bins));
        printf("\n");
};
        Hashmap_int hm ;
    Hashmap_int_data* data  = calloc(n_bytes_hashmap, 1);
        hm.n=n;
    hm.bins=bins;
    hm.n_entries=0;
    hm.data=data;
    return hm;
}
void hashmap_int_free (Hashmap_int* h){
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" free hashmap: ");
        printf(" h->n=");
        printf(printf_dec_format(h->n), h->n);
        printf(" (%s)", type_string(h->n));
        printf(" h->bins=");
        printf(printf_dec_format(h->bins), h->bins);
        printf(" (%s)", type_string(h->bins));
        printf(" h->n_entries=");
        printf(printf_dec_format(h->n_entries), h->n_entries);
        printf(" (%s)", type_string(h->n_entries));
        printf("\n");
};
        free(h->data);
            h->data=NULL;
}
Hashmap_int_pair hashmap_int_get (Hashmap_int* h, uint64_t key, int bin){
        // return key, value and pointer to value in data array
        // empty entry will have value.count==0
        assert(bin<h->bins);
            __auto_type limit_key  = key%h->n;
    __auto_type idx  = ((bin)+(((h->bins)*(limit_key))));
    __auto_type value  = h->data[idx];
    __auto_type valuep  = &(h->data[idx]);
        __auto_type p  = (Hashmap_int_pair) {key, value, valuep};
    return p;
}
Hashmap_int_pair hashmap_int_search (Hashmap_int* h, uint64_t key){
        for (int bin = 0;bin<h->bins;(bin)+=(1)) {
                        __auto_type p  = hashmap_int_get(h, key, bin);
        if ( 0<p.value.count ) {
                                    if ( (p.value.hash)==(key) ) {
                                                return p;
} else {
                                ;
};
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
                printf(" no entry: ");
                printf("\n");
};
            return (Hashmap_int_pair) {key, p.value, p.valuep};
};
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
        printf(" bin full: ");
        printf(" key=");
        printf(printf_dec_format(key), key);
        printf(" (%s)", type_string(key));
        printf("\n");
};
        return (Hashmap_int_pair) {key, (Hashmap_int_data) {0, 0, 0}, NULL};
}
bool hashmap_int_set (Hashmap_int* h, uint64_t key, int newvalue){
        // returns true if hashmap bin was empty (value -1)
        // return false if the vertex has been stored elsewhere
        // returns false if hashmap and all bins already contains a values with different hashes
        for (int bin = 0;bin<h->bins;(bin)+=(1)) {
                        __auto_type p  = hashmap_int_get(h, key, bin);
        if ( 0<p.value.count ) {
                        if ( (p.value.hash)==(key) ) {
                                                                __auto_type dat  = p.valuep;
                (dat->count)++;
                return false;
} else {
                                ;
}
} else {
                                                __auto_type dat  = p.valuep;
                        dat->value=newvalue;
            dat->hash=key;
            (dat->count)++;
            (h->n_entries)++;
            return true;
};
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
        printf(" hashmap collision: ");
        printf(" key=");
        printf(printf_dec_format(key), key);
        printf(" (%s)", type_string(key));
        printf("\n");
};
        return false;
}
bool equalp_Vertex (Vertex* a, Vertex* b){
        return (((a->pos[0])==(b->pos[0]))&&((a->pos[1])==(b->pos[1]))&&((a->pos[2])==(b->pos[2]))&&((a->texCoord[0])==(b->texCoord[0]))&&((a->texCoord[1])==(b->texCoord[1]))&&((a->color[0])==(b->color[0]))&&((a->color[1])==(b->color[1]))&&((a->color[2])==(b->color[2])));
}
int next_power_of_two (int n){
            __auto_type power  = 1;
    while (power<n) {
                power*=(2);
}
    return power;
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
            state._num_vertices=attrib.num_faces;
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
            state._num_indices=attrib.num_faces;
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
        __auto_type hashmap  = hashmap_int_make(attrib.num_faces, 8);
    __auto_type count_unique  = 0;
    // hashmap for vertex deduplication
    for (int i = 0;i<attrib.num_faces;(i)+=(1)) {
                        __auto_type v0  = attrib.vertices[((0)+(((3)*(attrib.faces[i].v_idx))))];
        __auto_type v1  = attrib.vertices[((1)+(((3)*(attrib.faces[i].v_idx))))];
        __auto_type v2  = attrib.vertices[((2)+(((3)*(attrib.faces[i].v_idx))))];
        __auto_type t0  = attrib.texcoords[((0)+(((2)*(attrib.faces[i].vt_idx))))];
        __auto_type t1  = attrib.texcoords[((1)+(((2)*(attrib.faces[i].vt_idx))))];
        __auto_type vertex  = (Vertex) {{v0, v1, v2}, {(1.e+0f), (1.e+0f), (1.e+0f)}, {t0, (-(t1))}};
        __auto_type key  = hash_Vertex(&vertex);
        if ( (true)==(hashmap_int_set(&hashmap, key, count_unique)) ) {
                                                state._vertices[count_unique]=vertex;
            state._indices[i]=count_unique;
            (count_unique)++;
} else {
                                                __auto_type p  = hashmap_int_search(&hashmap, key);
            if ( (0)==(p.value.count) ) {
                                                {
                                                            __auto_type current_time  = now();
                    printf("%6.6f", ((current_time)-(state._start_time)));
                    printf(" ");
                    printf(printf_dec_format(__FILE__), __FILE__);
                    printf(":");
                    printf(printf_dec_format(__LINE__), __LINE__);
                    printf(" ");
                    printf(printf_dec_format(__func__), __func__);
                    printf(" key not found: ");
                    printf(" key=");
                    printf(printf_dec_format(key), key);
                    printf(" (%s)", type_string(key));
                    printf(" i=");
                    printf(printf_dec_format(i), i);
                    printf(" (%s)", type_string(i));
                    printf(" count_unique=");
                    printf(printf_dec_format(count_unique), count_unique);
                    printf(" (%s)", type_string(count_unique));
                    printf(" p.value=");
                    printf(printf_dec_format(p.value), p.value);
                    printf(" (%s)", type_string(p.value));
                    printf("\n");
};
} else {
                                                                state._indices[i]=p.value.value;
                                __auto_type p  = hashmap_int_search(&hashmap, key);
                __auto_type vertex0  = state._vertices[p.value.value];
                if ( !(equalp_Vertex(&(vertex0), &vertex)) ) {
                                                            {
                                                                        __auto_type current_time  = now();
                        printf("%6.6f", ((current_time)-(state._start_time)));
                        printf(" ");
                        printf(printf_dec_format(__FILE__), __FILE__);
                        printf(":");
                        printf(printf_dec_format(__LINE__), __LINE__);
                        printf(" ");
                        printf(printf_dec_format(__func__), __func__);
                        printf(" collision: ");
                        printf(" ((vertex.pos[0])-(vertex0.pos[0]))=");
                        printf(printf_dec_format(((vertex.pos[0])-(vertex0.pos[0]))), ((vertex.pos[0])-(vertex0.pos[0])));
                        printf(" (%s)", type_string(((vertex.pos[0])-(vertex0.pos[0]))));
                        printf(" ((vertex.pos[1])-(vertex0.pos[1]))=");
                        printf(printf_dec_format(((vertex.pos[1])-(vertex0.pos[1]))), ((vertex.pos[1])-(vertex0.pos[1])));
                        printf(" (%s)", type_string(((vertex.pos[1])-(vertex0.pos[1]))));
                        printf(" ((vertex.pos[2])-(vertex0.pos[2]))=");
                        printf(printf_dec_format(((vertex.pos[2])-(vertex0.pos[2]))), ((vertex.pos[2])-(vertex0.pos[2])));
                        printf(" (%s)", type_string(((vertex.pos[2])-(vertex0.pos[2]))));
                        printf(" ((vertex.texCoord[0])-(vertex0.texCoord[0]))=");
                        printf(printf_dec_format(((vertex.texCoord[0])-(vertex0.texCoord[0]))), ((vertex.texCoord[0])-(vertex0.texCoord[0])));
                        printf(" (%s)", type_string(((vertex.texCoord[0])-(vertex0.texCoord[0]))));
                        printf(" ((vertex.texCoord[1])-(vertex0.texCoord[1]))=");
                        printf(printf_dec_format(((vertex.texCoord[1])-(vertex0.texCoord[1]))), ((vertex.texCoord[1])-(vertex0.texCoord[1])));
                        printf(" (%s)", type_string(((vertex.texCoord[1])-(vertex0.texCoord[1]))));
                        printf(" ((vertex.color[0])-(vertex0.color[0]))=");
                        printf(printf_dec_format(((vertex.color[0])-(vertex0.color[0]))), ((vertex.color[0])-(vertex0.color[0])));
                        printf(" (%s)", type_string(((vertex.color[0])-(vertex0.color[0]))));
                        printf(" ((vertex.color[1])-(vertex0.color[1]))=");
                        printf(printf_dec_format(((vertex.color[1])-(vertex0.color[1]))), ((vertex.color[1])-(vertex0.color[1])));
                        printf(" (%s)", type_string(((vertex.color[1])-(vertex0.color[1]))));
                        printf(" ((vertex.color[2])-(vertex0.color[2]))=");
                        printf(printf_dec_format(((vertex.color[2])-(vertex0.color[2]))), ((vertex.color[2])-(vertex0.color[2])));
                        printf(" (%s)", type_string(((vertex.color[2])-(vertex0.color[2]))));
                        printf(" hash_Vertex(&vertex)=");
                        printf(printf_dec_format(hash_Vertex(&vertex)), hash_Vertex(&vertex));
                        printf(" (%s)", type_string(hash_Vertex(&vertex)));
                        printf(" hash_Vertex(&vertex0)=");
                        printf(printf_dec_format(hash_Vertex(&vertex0)), hash_Vertex(&vertex0));
                        printf(" (%s)", type_string(hash_Vertex(&vertex0)));
                        printf("\n");
};
};
};
};
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
        printf(" hashmap finished: ");
        printf(" hashmap.n=");
        printf(printf_dec_format(hashmap.n), hashmap.n);
        printf(" (%s)", type_string(hashmap.n));
        printf(" hashmap.bins=");
        printf(printf_dec_format(hashmap.bins), hashmap.bins);
        printf(" (%s)", type_string(hashmap.bins));
        printf(" hashmap.n_entries=");
        printf(printf_dec_format(hashmap.n_entries), hashmap.n_entries);
        printf(" (%s)", type_string(hashmap.n_entries));
        printf(" count_unique=");
        printf(printf_dec_format(count_unique), count_unique);
        printf(" (%s)", type_string(count_unique));
        printf("\n");
};
    hashmap_int_free(&hashmap);
    {
                        __auto_type n_bytes_realloc  = ((count_unique)*(sizeof(*(state._vertices))));
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" realloc vertices: ");
            printf(" count_unique=");
            printf(printf_dec_format(count_unique), count_unique);
            printf(" (%s)", type_string(count_unique));
            printf(" n_bytes_realloc=");
            printf(printf_dec_format(n_bytes_realloc), n_bytes_realloc);
            printf(" (%s)", type_string(n_bytes_realloc));
            printf("\n");
};
                state._vertices=realloc(state._vertices, n_bytes_realloc);
        state._num_vertices=count_unique;
};
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