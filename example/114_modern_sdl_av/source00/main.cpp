#include "FancyWindow.h"

int main(int argc, char **argv) {
  auto w = FancyWindow({.Width = 320, .Height = 240});
  w.updateFrom();
  while (true) {
    w.present();
    SDL_Delay(3000);
  }

  return 0;
}
