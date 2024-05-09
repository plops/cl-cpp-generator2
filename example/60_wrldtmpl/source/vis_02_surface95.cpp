
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation

static char s_Font[51][5][6];
static bool fontInitialized = false;
static int s_Transl[256];
Surface::Surface(int w, int h, uint *a_Buffer)
    : buffer{b}, width{w}, height{h} {}
Surface::Surface(int w, int h) {
  (buffer) = (static_cast<uint *>(MALLOC64((w) * (h) * (sizeof(uint)))));
}
Surface::Surface(const char *file) : buffer{0}, width{0}, height{0} {
  auto f{fopen(file, "rb")};
  if (!(f)) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("file not found") << (" ") << (std::setw(8))
                  << (" file='") << (file) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  fclose(f);
  LoadImage(file);
}
Surface::~Surface() { FREE64(buffer); }
void Surface::InitCharSet() {
  SetChar(0, ":ooo:", "o:::o", "ooooo", "o:::o", "o:::o");
  SetChar(1, "oooo:", "o:::o", "oooo:", "o:::o", "oooo:");
  SetChar(2, ":oooo", "o::::", "o::::", "o::::", ":oooo");
  SetChar(3, "oooo:", "o:::o", "o:::o", "o:::o", "oooo:");
  SetChar(4, "ooooo", "o::::", "oooo:", "o::::", "ooooo");
  SetChar(5, "ooooo", "o::::", "ooo::", "o::::", "o::::");
  SetChar(6, ":oooo", "o::::", "o:ooo", "o:::o", ":ooo:");
  SetChar(7, "o:::o", "o:::o", "ooooo", "o:::o", "o:::o");
  SetChar(8, "::o::", "::o::", "::o::", "::o::", "::o::");
  SetChar(9, ":::o:", ":::o:", ":::o:", ":::o:", "ooo::");
  SetChar(10, "o::o:", "o:o::", "oo:::", "o:o::", "o::o:");
  SetChar(11, "o::::", "o::::", "o::::", "o::::", "ooooo");
  SetChar(12, "oo:o:", "o:o:o", "o:o:o", "o:::o", "o:::o");
  SetChar(13, "o:::o", "oo::o", "o:o:o", "o::oo", "o:::o");
  SetChar(14, ":ooo:", "o:::o", "o:::o", "o:::o", ":ooo:");
  SetChar(15, "oooo:", "o:::o", "oooo:", "o::::", "o::::");
  SetChar(16, ":ooo:", "o:::o", "o:::o", "o::oo", ":oooo");
  SetChar(17, "oooo:", "o:::o", "oooo:", "o:o::", "o::o:");
  SetChar(18, ":oooo", "o::::", ":ooo:", "::::o", "oooo:");
  SetChar(19, "ooooo", "::o::", "::o::", "::o::", "::o::");
  SetChar(20, "o:::o", "o:::o", "o:::o", "o:::o", ":oooo");
  SetChar(21, "o:::o", "o:::o", ":o:o:", ":o:o:", "::o::");
  SetChar(22, "o:::o", "o:::o", "o:o:o", "o:o:o", ":o:o:");
  SetChar(23, "o:::o", ":o:o:", "::o::", ":o:o:", "o:::o");
  SetChar(24, "o:::o", "o:::o", ":oooo", "::::o", ":ooo:");
  SetChar(25, "ooooo", ":::o:", "::o::", ":o:::", "ooooo");
  SetChar(26, ":ooo:", "o::oo", "o:o:o", "oo::o", ":ooo:");
  SetChar(27, "::o::", ":oo::", "::o::", "::o::", ":ooo:");
  SetChar(28, ":ooo:", "o:::o", "::oo:", ":o:::", "ooooo");
  SetChar(29, "oooo:", "::::o", "::oo:", "::::o", "oooo:");
  SetChar(30, "o::::", "o::o:", "ooooo", ":::o:", ":::o:");
  SetChar(31, "ooooo", "o::::", "oooo:", "::::o", "oooo:");
  SetChar(32, ":oooo", "o::::", "oooo:", "o:::o", ":ooo:");
  SetChar(33, "ooooo", "::::o", ":::o:", "::o::", "::o::");
  SetChar(34, ":ooo:", "o:::o", ":ooo:", "o:::o", ":ooo:");
  SetChar(35, ":ooo:", "o:::o", ":oooo", "::::o", ":ooo:");
  SetChar(36, "::o::", "::o::", "::o::", ":::::", "::o::");
  SetChar(37, ":ooo:", "::::o", ":::o:", ":::::", "::o::");
  SetChar(38, ":::::", ":::::", "::o::", ":::::", "::o::");
  SetChar(39, ":::::", ":::::", ":ooo:", ":::::", ":ooo:");
  SetChar(40, ":::::", ":::::", ":::::", ":::o:", "::o::");
  SetChar(41, ":::::", ":::::", ":::::", ":::::", "::o::");
  SetChar(42, ":::::", ":::::", ":ooo:", ":::::", ":::::");
  SetChar(43, ":::o:", "::o::", "::o::", "::o::", ":::o:");
  SetChar(44, "::o::", ":::o:", ":::o:", ":::o:", "::o::");
  SetChar(45, ":::::", ":::::", ":::::", ":::::", ":::::");
  SetChar(46, "ooooo", "ooooo", "ooooo", "ooooo", "ooooo");
  SetChar(47, "::o::", "::o::", ":::::", ":::::", ":::::");
  SetChar(48, "o:o:o", ":ooo:", "ooooo", ":ooo:", "o:o:o");
  SetChar(49, "::::o", ":::o:", "::o::", ":o:::", "o::::");
  char c[49]{"abcdefghijklmnopqrstuvwxyz0123456789!?:=.-() #'*/"};
  for (auto i = 0; (i) < (256); (i) += (1)) {
    ((s_Transl)[(i)]) = (45);
  }
  for (auto i = 0; (i) < (49); (i) += (1)) {
    ((s_Transl)[(static_cast<unsigned char>((c)[(i)]))]) = (i);
  }
}
void Surface::SetChar(const char *c1, const char *c2, const char *c3,
                      const char *c4, const char *c5) {
  strcpy((s_Font)[(c)][(1)], c1);
  strcpy((s_Font)[(c)][(2)], c2);
  strcpy((s_Font)[(c)][(3)], c3);
  strcpy((s_Font)[(c)][(4)], c4);
  strcpy((s_Font)[(c)][(5)], c5);
}
void Surface::Print(const char *tt, int x1, int y1, uint c) {
  if (!(fontInitialized)) {
    InitCharset();
    (fontInitialized) = (true);
  }
  auto tt{(buffer) + (x1) + ((y1) * (width))};
  for (auto i = 0; (i) < (static_cast<int>(strlen(s))); (i) += (1)) {
    auto pos{0};
    if (() <= () && () <= ()) {
      (pos) = ((s_Transl)[(
          static_cast<unsigned short>(((s)[(i)]) - (('A') - ('a'))))]);
    } else {
      (pos) = ((s_Transl)[(static_cast<unsigned short>((s)[(i)]))]);
    }
    auto a{tt};
    auto u{static_cast<const char *>((s_Font)[(pos)])};
    for (auto v = 0; (v) < (5); (v) += (1)) {
      for (auto h = 0; (h) < (5); (h) += (1)) {
        if (('o') == (*u++)) {
          (*((a) + (h))) = (c);
          (*((a) + (h) + (width))) = (0);
        }
      }
      (u)++;
      (a) += (width);
    }
    (tt) += (6);
  }
}
void Surface::Clear(uint c) {
  const int s{(width) * (height)};
  for (auto i = 0; (i) < (s); (i) += (1)) {
    ((buffer)[(i)]) = (c);
  }
}
void Surface::LoadImage(const char *file) {
  FREE_IMAGE_FORMAT fif{FIF_UNKNOWN};
  (fif) = (FreeImage_GetFileType(file, 0));
  if ((FIF_UNKNOWN) == (fif)) {
    (fif) = (FreeImage_GetFIFFromFilename(file));
  }
  auto tmp{FreeImage_Load(fif, file)};
  auto dib{FreeImage_ConvertTo32Bits(tmp)};
  FreeImage_Unload(tmp);
  auto width{FreeImage_GetWidth(dib)};
  auto height{FreeImage_GetHeight(dib)};
  auto buffer{
      static_cast<uint *>(MALLOC64((width) * (height) * (sizeof(uint))))};
  for (auto y = 0; (y) < (height); (y) += (1)) {
    auto line{FreeImage_GetScanLine(dib, (height) + (-1) + (-y))};
    memcpy((buffer) + ((y) * (width)), line, (width) * (sizeof(uint)));
  }
  FreeImage_Unload(dib);
}
void Surface::CopyTo(Surface *dst, int a_X, int a_Y) {
  auto dst{d->buffer};
  auto src{buffer};
  if ((src) & (dst)) {
    auto sw{width};
    auto sh{height};
    auto w{d->width};
    auto h{d->height};
    if ((w) < ((sw) + (x))) {
      (sw) = ((w) - (x));
    }
    if ((h) < ((sh) + (y))) {
      (sh) = ((h) - (y));
    }
    if ((x) < (0)) {
      (src) -= (x);
      (sw) += (x);
      (x) = (0);
    }
    if ((y) < (0)) {
      (src) -= ((sw) * (y));
      (sh) += (y);
      (y) = (0);
    }
    if (((0) < (sw)) & ((0) < (sh))) {
      (dst) += ((x) + ((w) * (y)));
      for (auto y = 0; (y) < (sh); (y) += (1)) {
        memcpy(dst, src, (4) * (sw));
        (dst) += (w);
        (src) += (sw);
      }
    }
  }
}
void Surface::Box(int x1, int y1, int x2, int y2, uint color) {}
void Surface::Bar(int x1, int y1, int x2, int y2, uint color) {}