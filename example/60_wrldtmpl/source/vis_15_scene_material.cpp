
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

Material::Material() {
  nil;
  (m_Texture) = (0);
  (m_Name) = (new (char)[(64)]);
  SetDiffuse(make_float3(0.80F, 0.80F, 0.80F));
  nil;
}
Material::~Material() {
  nil;
  delete (m_Name);
  nil;
}
void Material::SetDiffuse(const float3 &a_Diff) {
  nil;
  (m_Diff) = (a_Diff);
  nil;
}
void Material::SetTexture(const Texture *a_Texture) {
  nil;
  (m_Texture) = (static_cast<Texture *>(a_Texture));
  nil;
}
void Material::SetName(char *a_Name) {
  nil;
  strcpy(m_Name, a_Name);
  nil;
}
void Material::SetIdx(uint a_Idx) {
  nil;
  (m_Idx) = (a_Idx);
  nil;
}
const float3 Material::GetDiffuse() {
  nil;
  return m_Diff;
  nil;
}
const Texture *Material::GetTexture() {
  nil;
  return m_Texture;
  nil;
}
char *Material::GetName() {
  nil;
  return m_Name;
  nil;
}
uint Material::GetIdx() {
  nil;
  return m_Idx;
  nil;
}