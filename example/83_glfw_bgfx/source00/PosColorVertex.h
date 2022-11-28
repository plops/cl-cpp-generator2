#ifndef POSCOLORVERTEX_H
#define POSCOLORVERTEX_H

#include <bgfx/bgfx.h>
class PosColorVertex  {
        public:
        float m_x;
        float m_y;
        float m_z;
        uint32_t m_abgr;
        static bgfx::VertexLayout ms_decl;
        static void init ()     ;  
};

#endif /* !POSCOLORVERTEX_H */