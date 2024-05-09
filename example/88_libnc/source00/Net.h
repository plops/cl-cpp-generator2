#ifndef NET_H
#define NET_H

struct NCContext; 
struct NCDevice; 
class Net  {
        public:
        NCContext *s; 
        NCDevice *dev; 
         Net ()       ;   
         ~Net ()       ;   
};

#endif /* !NET_H */