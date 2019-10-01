
# vulkan_01_instance.c:43:6: warning: no previous declaration for 'createInstance' [-Wmissing-declarations]
#    43 | void createInstance() {
#       |      ^~~~~~~~~~~~~~

# gcc -std=c18 -c vulkan_01_instance.c -Wmissing-declarations 2>&1|grep '{'|cut -d '|' -f 2|cut -d '{' -f 1
# void createInstance() 

gcc `pkg-config --cflags cglm` -std=gnu18 -S source/vulkan_*.c -Wmissing-declarations 2>&1|grep "warning: no previous declaration for.*-Wmissing-declarations" -A1|grep '{'|cut -d '|' -f 2|cut -d '{' -f 1|awk '{print $N";"}' > source/proto.h

