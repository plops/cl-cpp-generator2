#ifndef VIS_16_SCENE_MATMANAGER_H
#define VIS_16_SCENE_MATMANAGER_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class MatManager {
public:
  MatManager();
  void LoadMTL(char *a_File);
  Material *FindMaterial(char *a_Name);
  Material *GetMaterial(int a_Idx);
  void Reset();
  void AddMaterial(Material *a_Mat);
  uint GetMatCount();

private:
  Material **m_Mat;
  uint m_NrMat;
};
#endif