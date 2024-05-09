#ifndef VIS_14_SCENE_TEXTURE_H
#define VIS_14_SCENE_TEXTURE_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Texture {
public:
  Texture();
  void Init(char *a_File);
  const unsigned int *GetBitmap();
  const unsigned int GetWidth();
  const unsigned int GetHeight();
  unsigned int *m_B32;
  unsigned int m_Width, m_Height;
};
#endif