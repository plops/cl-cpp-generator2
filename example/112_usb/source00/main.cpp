#include "usbpp.hpp"

int main(int argc, char **argv) {
  auto ctx = init();
  auto devices = get_device_list(ctx);
  auto bt = find_if(devices, [](auto &dev) {});
}
