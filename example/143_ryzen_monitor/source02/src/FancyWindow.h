#ifndef FANCYWINDOW_H
#define FANCYWINDOW_H

#include <SDL2/SDL.h>
#include "c_resource.hpp"
using Window=c_resource<SDL_Window,SDL_CreateWindow,SDL_DestroyWindow>;
using Renderer=c_resource<SDL_Renderer,SDL_CreateRenderer,SDL_DestroyRenderer>;
using Texture=c_resource<SDL_Texture,SDL_CreateTexture,SDL_DestroyTexture>;

struct tDimensions {
        uint16_t Width;
        uint16_t Height;
};


static constexpr bool successful (int Code)      ;  

static auto centeredBox (tDimensions Dimensions, int Monitor = SDL_GetNumVideoDisplays())   noexcept   ;  
class FancyWindow  {
        public:
        explicit  FancyWindow (tDimensions Dimensions)   noexcept    ;  
        void updateFrom ()   noexcept    ;  
        void present ()   noexcept    ;  
        public:
        Window Window_;
        Renderer Renderer_;
        Texture Texture_;
        int Width_, Height_, PixelsPitch_, SourceFormat_;
};

bool isAlive ()   noexcept   ;  

#endif /* !FANCYWINDOW_H */