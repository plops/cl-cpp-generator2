#pragma once
#include "UsbError.h"
#include "UsbUsing.h"
#include <libusb-1.0/libusb.h>
#include <vector>

context init() {
  libusb_context *ctx = nullptr;
  check(libusb_init(&ctx));
  return context{ctx};
}

std::vector<device> get_device_list(context &ctx) {
  libusb_device **list = nullptr;
  auto n = libusb_get_device_list(ctx.get(), &list);
  check(n);
  auto ret = std::vector<device>();
  for (auto i = 0; i < n; i += 1) {
    ret.emplace_back(list[i]);
  }
  libusb_free_device_list(list, false);
  return ret;
}

device_handle open(device &dev) {
  libusb_device_handle *handle = nullptr;
  auto err = libusb_open(dev.get(), &handle);
  check(err);
  return device_handle{handle};
}

device_descriptor get_device_descriptor(const device &dev) {
  auto ret = device_descriptor();
  check(libusb_get_device_descriptor(dev.get(), &ret));
  return ret;
}

device_handle open_device_with_vid_pid(context &ctx, uint16_t vid,
                                       uint16_t pid) {
  auto h = libusb_open_device_with_vid_pid(ctx.get(), vid, pid);
  device_handle ret{h};
  if (nullptr == ret) {
    throw UsbError(LIBUSB_ERROR_NOT_FOUND);
  }
  return ret;
}
