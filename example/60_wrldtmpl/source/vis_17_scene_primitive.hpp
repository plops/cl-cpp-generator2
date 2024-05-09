#ifndef VIS_17_SCENE_PRIMITIVE_H
#define VIS_17_SCENE_PRIMITIVE_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Primitive {
public:
  Primitive();
  void Init(Vertex *a_V1, Vertex *a_V2, Vertex *a_V3);
  const Material *GetMaterial();
  void SetMaterial(const Material *a_Mat);
  void SetNormal(const float3 &a_N);
  void UpdateNormal();
  const Vertex *GetVertex(const uint a_Idx);
  void SetVertex(const uint a_Idx, Vertex *a_Vertex);
  const float3 GetNormal(float u, float v);
  float3 m_N;
  Vertex *m_Vertex[3];
  Material *m_Material;
};
#endif