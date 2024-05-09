
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

Texture::Texture() {
  nil;
  nil;
  nil;
}
void Texture::Init(char *a_File) {
  nil;
  (m_B32) = (0);
  auto f{fopen(a_File, "rb")};
  if (!(f)) {
    return;
  }
  auto fif{FIF_UNKNOWN};
  (fif) = (FreeImage_GetFileType(a_File, 0));
  if ((FIF_UNKNOWN) == (fif)) {
    (fif) = (FreeImage_GetFIFFromFilename(a_File));
  }
  auto tmp{FreeImage_Load(fif, a_File)};
  auto dib{FreeImage_ConvertTo32Bits(tmp)};
  FreeImage_Unload(tmp);
  auto bits{FreeImage_GetBits(dib)};
  (m_Width) = (FreeImage_GetWidth(dib));
  (m_Height) = (FreeImage_GetHeight(dib));
  (m_B32) =
      (static_cast<uint *>(MALLOC64((m_Width) * (m_Height) * (sizeof(uint)))));
  for (auto y = 0; (y) < (m_Height); (y) += (1)) {
    auto line{FreeImage_GetScanLine(dib, (m_Height) - (1) - (y))};
    memcpy((m_B32) + ((y) * (m_Width)), line, (m_Width) * (sizeof(uint)));
  }
  FreeImage_Unload(dib);
  nil;
}
const unsigned int *Texture::GetBitmap() {
  nil;
  return m_B32;
  nil;
}
const unsigned int Texture::GetWidth() {
  nil;
  return m_Width;
  nil;
}
const unsigned int Texture::GetHeight() {
  nil;
  return m_Height;
  nil;
}