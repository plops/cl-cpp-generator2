
# vulkan_01_instance.c:43:6: warning: no previous declaration for 'createInstance' [-Wmissing-declarations]
#    43 | void createInstance() {
#       |      ^~~~~~~~~~~~~~

# gcc -std=c18 -c vulkan_01_instance.c -Wmissing-declarations 2>&1|grep '{'|cut -d '|' -f 2|cut -d '{' -f 1
# void createInstance() 


gcc -std=c18 -c source/vulkan_*.c -Wmissing-declarations 2>&1|grep '{'|cut -d '|' -f 2|cut -d '{' -f 1|awk '{print $N";"}' > source/proto.h
