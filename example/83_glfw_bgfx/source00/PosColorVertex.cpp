// no preamble
;
#include "PosColorVertex.h"
#include <bgfx/bgfx.h>
void PosColorVertex::init() {
  ms_decl.begin()
      .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
      .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
      .end();
}