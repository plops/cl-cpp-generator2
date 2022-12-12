#ifndef APP_H
#define APP_H

#include "bla.h"
class App  {
        public:
        ovrJava* java;
        bool resumed;
        Egl egl;
        Renderer renderer;
        ANativeWindow* window;
        ovrMobile* ovr;
        bool back_button_down_previous_frame;
        uint64_t frame_index;
         App (ovrJava* java)     ;  
};

#endif /* !APP_H */