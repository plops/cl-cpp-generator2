
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

aabb::aabb() : bmin(make_float3(1.00e+3F, 1.00e+3F, 1.00e+3F)) {
  bmax(make_float3(-1.00e+3F, -1.00e+3F, -1.00e+3F))
}
{}
Scene::Scene() {
  nil;
  (m_MatMan) = (new MatManager);
  nil;
}
void Scene::InitSceneState() {
  nil;
  (m_Extends.bmin) = (make_float3(1.00e+3F, 1.00e+3F, 1.00e+3F));
  (m_Extends.bmax) = (make_float3(-1.00e+3F, -1.00e+3F, -1.00e+3F));
  (scene) = (this);
  nil;
}
bool Scene::InitScene(const char *a_File) {
  nil;
  InitSceneState();
  (path) = (new (char)[(1024)]);
  (file) = (new (char)[(1024)]);
  (noext) = (new (char)[(1024)]);
  strcpy(path, a_File);
  auto pos{path};
  while (strstr(pos, "/")) {
    (pos) = ((strstr(pos, "/")) + (1));
  }
  while (strstr(pos, "\\")) {
    (pos) = ((strstr(pos, "\\")) + (1));
  }
  (*pos) = (0);
  (pos) = (static_cast<char *>(a_File));
  while (strstr(pos, "/")) {
    (pos) = ((strstr(pos, "/")) + (1));
  }
  while (strstr(pos, "\\")) {
    (pos) = ((strstr(pos, "\\")) + (1));
  }
  if ((strstr(a_File, "/")) | (strstr(a_File, "\\"))) {
    strcpy(file, pos);
  } else {
    strcpy(file, a_File);
  }
  LoadOBJ(a_File);
  strcpy(noext, file);
  (pos) = (noext);
  while (strstr(pos, ".")) {
    (pos) = (strstr(pos, "."));
  }
  if ((*pos) == (".")) {
    (*pos) = (0);
  }
  return true;
  nil;
}
const aabb &Scene::GetExtends() {
  nil;
  return m_Extends;
  nil;
}
void Scene::SetExtends(aabb a_Box) {
  nil;
  (m_Extends) = (a_Box);
  nil;
}
MatManager *Scene::GetMatManager() {
  nil;
  return m_MatMan;
  nil;
}
void Scene::UpdateSceneExtends() {
  nil;
  auto emin{make_float3(10000, 10000, 10000)};
  auto emax{make_float3(-10000, -10000, -10000)};
  for (auto j = 0; (j) < (m_Primitives); (j) += (1)) {
    for (auto v = 0; (v) < (3); (v) += (1)) {
      auto pos{(m_Prim)[(j)].GetVertex(v).->GetPos()};
      for (auto a = 0; (a) < (3); (a) += (1)) {
        if (((pos.e)[(a)]) < (emin.e(a))) {
          ((emin.e)[(a)]) = ((pos.e)[(a)]);
        }
        if ((emax.e(a)) < ((pos.e)[(a)])) {
          ((emax.e)[(a)]) = ((pos.e)[(a)]);
        }
      }
    }
  }
  (m_Extends.bmin) = (emin);
  (m_Extends.bmax) = (emax);
  nil;
}
void Scene::LoadOBJ(const char *filename) {
  nil;
  auto f{fopen(filename, "r")};
  if (!(f)) {
    return;
  }
  auto fcount{static_cast<uint>(0)};
  auto ncount{static_cast<uint>(0)};
  auto uvcount{static_cast<uint>(0)};
  auto vcount{static_cast<uint>(0)};
  char buffer[256];
  char cmd[256];
  char objname[256];
  while (true) {
    fgets(buffer, 250, f);
    if (feof(f)) {
      break;
    }
    if (('v') == ((buffer)[(0)])) {
      switch ((buffer)[(1)]) {
      case ' ': {
        (vcount)++;
        break;
      };
      case 't': {
        (uvcount)++;
        break;
      };
      case 'n': {
        (ncount)++;
        break;
      };
      }
    } else {
      if ((('f') == ((buffer)[(0)])) & ((' ') == ((buffer)[(1)]))) {
        (fcount)++;
      }
    }
  }
  fclose(f);
  (m_Prim) =
      (static_cast<Primitive *>(MALLOC64((fcount) * (sizeof(Primitive)))));
  (f) = (fopen(filename, "r"));
  auto curmat{static_cast<Material *>(nullptr)};
  auto verts{static_cast<uint>(0)};
  auto uvs{static_cast<uint>(0)};
  auto normals{static_cast<uint>(0)};
  auto vert{new (float3)[(vcount)]};
  auto norm{new (float3)[(ncount)]};
  auto tu{new (float)[(uvcount)]};
  auto tv{new (float)[(uvcount)]};
  auto vertex{static_cast<Vertex *>(
      MALLOC64((sizeof(Vertex)) * ((4) + ((fcount) * (3)))))};
  auto vidx{static_cast<uint>(0)};
  char currobject[256];
  strcpy(currobj, "none");
  (m_Primitives) = (0);
  while (true) {
    fgets(buffer, 250, f);
    if (feof(f)) {
      break;
    }
    switch ((buffer)[(0)]) {
    case 'v': {
      switch ((buffer)[(1)]) {
      case ' ': {
        auto x{0.F};
        auto y{0.F};
        auto z{0.F};
        sscanf(buffer, "%s %f %f %f", cmd, &x, &y, &z);
        auto pos{make_float3(x, y, z)};
        (verts)++;
        ((vert)[(verts)]) = (pos);
        for (auto a = 0; (a) < (3); (a) += (1)) {
          if (((pos.e)[(a)]) < (m_Extends.bmin.e(a))) {
            ((m_Extends.bmin.e)[(a)]) = ((pos.e)[(a)]);
          }
          if ((m_Extends.bmax.e(a)) < ((pos.e)[(a)])) {
            ((m_Extends.bmax.e)[(a)]) = ((pos.e)[(a)]);
          }
        }
        break;
      };
      case 't': {
        auto u{0.F};
        auto v{0.F};
        sscanf(buffer, "%s %f %f", cmd, &u, &v);
        ((tu)[(uvs)]) = (u);
        (uvs)++;
        ((tv)[(uvs)]) = (-v);
        break;
      };
      case 'n': {
        auto x{0.F};
        auto y{0.F};
        auto z{0.F};
        sscanf(buffer, "%s %f %f %f", cmd, &x, &y, &z);
        (normals)++;
        ((norm)[(normals)]) = (make_float3(-x, -y, -z));
        break;
      };
      }
      break;
    };
    default: {
      break;
      break;
    };
    }
  }
  fclose(f);
  (f) = (fopen(filename, "r"));
  while (true) {
    fgets(buffer, 250, f);
    if (feof(f)) {
      break;
    }
    switch ((buffer)[(0)]) {
    case 'f': {
      Vertex *v[3];
      float cu[3];
      float cv[3];
      auto tex{curmat->GetTexture()};
      uint vnr[9];
      auto vars{sscanf((buffer) + (2), "%i/%i/%i %i/%i/%i %i/%i/%i",
                       &((vnr)[(0)]), &((vnr)[(1)]), &((vnr)[(2)]),
                       &((vnr)[(3)]), &((vnr)[(4)]), &((vnr)[(5)]),
                       &((vnr)[(6)]), &((vnr)[(7)]), &((vnr)[(8)]))};
      if ((vars) < (9)) {
        (vars) = (sscanf((buffer) + (2), "%i/%i %i/%i %i/%i", &((vnr)[(0)]),
                         &((vnr)[(2)]), &((vnr)[(3)]), &((vnr)[(5)]),
                         &((vnr)[(6)]), &((vnr)[(8)])));
        if ((vars) < (6)) {
          sscanf((buffer) + (2), "%i//%i %i//%i %i//%i", &((vnr)[(0)]),
                 &((vnr)[(2)]), &((vnr)[(3)]), &((vnr)[(5)]), &((vnr)[(6)]),
                 &((vnr)[(8)]));
        }
      }
      for (auto i = 0; (i) < (3); (i) += (1)) {
        (vidx)++;
        ((v)[(i)]) = ((&vertex)[(vidx)]);
        if (tex) {
          auto vidx{((vnr)[(((i) * (3)) + (1))]) - (1)};
          ((cu)[(i)]) = ((tu)[(vidx)]);
          ((cv)[(i)]) = ((tv)[(vidx)]);
        }
        auto nidx{((vnr)[((2) + ((i) * (3)))]) - (1)};
        auto vidx{((vnr)[((i) * (3))]) - (1)};
        ((v)[(i)])->(SetNormal((norm)[(nidx)]));
        ((v)[(i)])->(SetPos((vert)[(vidx)]));
      }
      (m_Primitives)++;
      auto p{&((m_Prim)[(m_Primitives)])};
      if (tex) {
        ((v)[(0)])->(SetUV(((cu)[(0)]) * (tex->m_Width()),
                           ((cv)[(0)]) * (tex->m_Height())));
        ((v)[(1)])->(SetUV(((cu)[(1)]) * (tex->m_Width()),
                           ((cv)[(1)]) * (tex->m_Height())));
        ((v)[(2)])->(SetUV(((cu)[(2)]) * (tex->m_Width()),
                           ((cv)[(2)]) * (tex->m_Height())));
      }
      p->Init((v)[(0)], (v)[(1)], (v)[(2)]);
      p->SetMaterial(curmat);
      break;
    };
    case 'g': {
      sscanf((buffer) + (2), "%s", objname);
      strcpy(currobj, objname);
      break;
    };
    case 'm': {
      if (!(_strnicmp(buffer, "mtllib", 6))) {
        char libname[128];
        char fullname[256];
        sscanf((buffer) + (7), "%s", libname);
        strcpy(fullname, path);
        strcat(fullname, libname);
        m_MatMan->LoadMTL(fullname);
      }
      break;
    };
    case 'u': {
      if (!(_strnicmp(buffer, "usemtl", 6))) {
        char matname[128];
        sscanf((buffer) + (7), "%s", matname);
        (curman) = (m_MatMan->FindMaterial(matname));
      }
      break;
    };
    }
  }
  nil;
}