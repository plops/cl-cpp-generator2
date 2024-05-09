#ifndef VIS_18_SCENE_H
#define VIS_18_SCENE_H
#include "utils.h"
;
#include "globals.h"
;
// header

static Scene *scene;
;
class aabb {
public:
  aabb();
  float3 bmin;
  float3 bmax;
};
class Scene {
public:
  Scene();
  void InitSceneState();
  bool InitScene(const char *a_File);
  const aabb &GetExtends();
  void SetExtends(aabb a_Box);
  MatManager *GetMatManager();
  void UpdateSceneExtends();
  void LoadOBJ(const char *filename);
  uint m_Primitives;
  uint m_MaxPrims;
  aabb m_Extends;
  MatManager *m_MatMan;
  Primitive *m_Prim;
  char *path;
  char *file;
  char *noext;
};
#endif