#ifndef APP_H
#define APP_H

#include "Egl.h"
#include "Renderer.h"
#include <iostream>
#include "VrApi.h"
#include "VrApi_Helpers.h"
#include "VrApi_Input.h"
#include "VrApi_SystemUtils.h"
#include "android_native_app_glue.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/window.h>
#include <vector>
#include <cstdlib>
#include <unistd.h>
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
        void update_vr_mode ()     ;  
        void handle_input ()     ;  
};

#endif /* !APP_H */