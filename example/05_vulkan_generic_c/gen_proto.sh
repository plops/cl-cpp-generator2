gcc -std=c18 -c source/vulkan_*.c -Wmissing-declarations 2>&1|grep '{'|cut -d '|' -f 2|cut -d '{' -f 1|awk '{print $N";"}' > source/proto.h
