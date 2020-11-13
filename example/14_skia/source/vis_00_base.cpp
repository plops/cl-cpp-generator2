
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <include/core/SkCanvas.h>
#include <include/core/SkGraphics.h>
#include <include/core/SkImageEncoder.h>
#include <include/core/SkString.h>
#include <iostream>
#include <thread>

// implementation
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto path = SkString("skhello.png");
  auto paint = SkPaint();
  paint.setARGB(255, 255, 255, 255);
  paint.setAntiAlias(true);
  auto width = SkScalar(800);
  auto height = SkScalar(600);
  auto bitmap = SkBitmap();
  bitmap.allocPixels(SkImageInfo::MakeN32Premul(width, height));
  auto canvas = SkCanvas(bitmap);
  canvas.drawColor(SK_ColorWHITE);
  auto src = SkEncodeBitmap(bitmap, SkEncodedImageFormat::kPNG, 100);
  auto img = src.get();
  return 0;
}