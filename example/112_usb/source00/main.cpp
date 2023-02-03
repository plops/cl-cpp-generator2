#include "usbpp.hpp"

int main(int argc, char **argv) {
  auto ctx = init();
  auto bt = open_device_with_vid_pid(ctx, 0x8087, 0x0a2b);
}
