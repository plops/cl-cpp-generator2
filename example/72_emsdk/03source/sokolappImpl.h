#ifndef SOKOLAPPIMPL_H
#define SOKOLAPPIMPL_H

#include <memory>
class sg_desc;
extern "C" struct sapp_desc;
extern "C" struct sapp_event;
class sokolappImpl  {
        public:
         sokolappImpl ()     ;  
         ~sokolappImpl ()     ;  
        // placeholder private-code-inside-class
;
};
extern "C" void init ()    ;  
extern "C" void frame ()    ;  
extern "C" void cleanup ()    ;  
extern "C" void input (const sapp_event* event)    ;  
extern "C" sapp_desc sokol_main (int argc, char** argv)    ;  

#endif /* !SOKOLAPPIMPL_H */