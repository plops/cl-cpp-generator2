#include "FancyWindow.h"

int main(int argc, char **argv) {
  auto w = FancyWindow({.Width = 320, .Height = 240});
  w.updateFrom();
  w.present();

  return 0;
}
