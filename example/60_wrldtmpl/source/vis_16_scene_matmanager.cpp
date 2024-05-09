
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

MatManager::MatManager() {
  nil;
  (m_Mat) = (new (Material *)[(1024)]);
  ((m_Mat)[(0)]) = (new Material());
  ((m_Mat)[(0)])->(SetName("DEFAULT"));
  (m_NrMat) = (1);
  nil;
}
void MatManager::LoadMTL(char *a_File) {
  nil;
  auto f{fopen(a_File, "r")};
  if (!(f)) {
    return;
  }
  auto curmat{static_cast<uint>(0)};
  char buffer[256];
  char cmd[128];
  while (!(feof(f))) {
    fgets(buffer, 250, f);
    sscanf(buffer, "%s", cmd);
    if (!(_stricmp(cmd, "newmtl"))) {
      (m_NrMat)++;
      (curmat) = (m_NrMat);
      ((m_Mat)[(curmat)]) = (new Material());
      char matname[128];
      sscanf((buffer) + (strlen(cmd)), "%s", matname);
      ((m_Mat)[(curmat)])->(SetName(matname));
    }
    if (!(_stricmp(cmd, "Kd"))) {
      auto r{0.F};
      auto g{0.F};
      auto b{0.F};
      sscanf((buffer) + (3), "%f %f %f", &r, &g, &b);
      auto c{make_float3(r, g, b)};
      ((m_Mat)[(curmat)])->(SetDiffuse(c));
    }
    if (!(_stricmp(cmd, "map_Kd"))) {
      char tname[128];
      char fname[128];
      ((tname)[(0)]) = (0);
      ((fname)[(0)]) = (0);
      sscanf((buffer) + (7), "%s", tname);
      if ((tname)[(0)]) {
        strcpy(fname, scene->path);
        if (!(strstr(tname, "textures/"))) {
          strcat(fname, "textures/");
        }
        strcat(fname, tname);
        auto tex{new Texture};
        tex->Init(fname);
        if (!(tex->m_B32())) {
          strcpy(fname, scene->path);
          strcat(fname, tname);
          tex->Init(fname);
        }
        ((m_Mat)[(curmat)])->(SetTexture(tex));
      }
    }
  }
  nil;
}
Material *MatManager::FindMaterial(char *a_Name) {
  nil;
  for (auto i = 0; (i) < (m_NrMat); (i) += (1)) {
    if (!(_stricmp(((m_Mat)[(i)])->(GetName()), a_Name))) {
      return (m_Mat)[(i)];
    }
  }
  return (m_Mat)[(0)];
  nil;
}
Material *MatManager::GetMaterial(int a_Idx) {
  nil;
  return (m_Mat)[(a_Idx)];
  nil;
}
void MatManager::Reset() {
  nil;
  (m_NrMat) = (0);
  nil;
}
void MatManager::AddMaterial(Material *a_Mat) {
  nil;
  (m_NrMat)++;
  ((m_Mat)[(m_NrMat)]) = (a_Mat);
  nil;
}
uint MatManager::GetMatCount() {
  nil;
  return m_NrMat;
  nil;
}