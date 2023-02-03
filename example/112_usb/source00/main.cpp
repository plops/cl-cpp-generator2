#include "usbpp.hpp"

int main(int argc, char **argv) {
  auto ctx = init();
  auto devices = get_device_list(ctx);
  auto bt = find_if(devices, [](auto dev) {
    auto d = get_device_descriptor(dev);
    return ((0x32903 == d.idVendor) & (0x5455 == d.idProduct));
  });
}
