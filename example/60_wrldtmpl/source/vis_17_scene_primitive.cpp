
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

Primitive::Primitive() {
  nil;
  nil;
  nil;
}
void Primitive::Init(Vertex *a_V1, Vertex *a_V2, Vertex *a_V3) {
  nil;
  ((m_Vertex)[(0)]) = (a_V1);
  ((m_Vertex)[(1)]) = (a_V2);
  ((m_Vertex)[(2)]) = (a_V3);
  UpdateNormal();
  nil;
}
const Material *Primitive::GetMaterial() {
  nil;
  return m_Material;
  nil;
}
void Primitive::SetMaterial(const Material *a_Mat) {
  nil;
  (m_Material) = (static_cast<Material *>(a_Mat));
  nil;
}
void Primitive::SetNormal(const float3 &a_N) {
  nil;
  (m_N) = (a_N);
  nil;
}
void Primitive::UpdateNormal() {
  nil;
  auto c{normalize((((m_Vertex)[(1)])->(GetPos())) -
                   (((m_Vertex)[(0)])->(GetPos())))};
  auto b{normalize((((m_Vertex)[(2)])->(GetPos())) -
                   (((m_Vertex)[(0)])->(GetPos())))};
  (m_N) = (cross(b, c));
  nil;
}
const Vertex *Primitive::GetVertex(const uint a_Idx) {
  nil;
  return (m_Vertex)[(a_Idx)];
  nil;
}
void Primitive::SetVertex(const uint a_Idx, Vertex *a_Vertex) {
  nil;
  ((m_Vertex)[(a_Idx)]) = (a_Vertex);
  nil;
}
const float3 Primitive::GetNormal(float u, float v) {
  nil;
  return m_N;
  nil;
}