#ifndef VIS_15_SCENE_MATERIAL_H
#define VIS_15_SCENE_MATERIAL_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Material {
public:
  Material();
  ~Material();
  void SetDiffuse(const float3 &a_Diff);
  void SetTexture(const Texture *a_Texture);
  void SetName(char *a_Name);
  void SetIdx(uint a_Idx);
  const float3 GetDiffuse();
  const Texture *GetTexture();
  char *GetName();
  uint GetIdx();

private:
  Texture *m_Texture;
  uint m_Idx;
  float3 m_Diff;
  char *m_Name;
};
#endif