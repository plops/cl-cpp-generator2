#pragma once
#include "UsbError.h"
#include "UsbUsing.h"
#include <chrono>
#include <iostream>
#include <libusb-1.0/libusb.h>
class UsbInterface {
  static constexpr int Invalid = -1;
  int handle = Invalid;
  libusb_device_handle *dev = nullptr;

  int try_release() {
    if (!(Invalid == handle)) {
      auto h = handle;
      handle = Invalid;

      return libusb_release_interface(dev, h);
    }
    return Invalid;
  }

public:
  UsbInterface(int i, device_handle &dev) : handle(i), dev(dev.get()) {}
  UsbInterface(const UsbInterface &) = delete;
  UsbInterface &operator=(const UsbInterface &) = delete;
  UsbInterface(UsbInterface &&from) { *this = std::move(from); }

  void release_interface() { check(try_release()); }

  UsbInterface &operator=(UsbInterface &&from) {
    release_interface();
    handle = from.handle;
    dev = from.dev;
    from.handle = Invalid;

    return *this;
  }

  ~UsbInterface() {
    auto e = libusb_release_interface(dev, handle);
    if (!(e == 0)) {
      std::cerr << "failed to release interface: ";
    }
  }
};
template <typename T, typename... Args>
constexpr bool one_of = (... || std::same_as<T, Args>);
template <typename T>
concept NonConstByteData =
    std::ranges::contiguous_range<T> &&
    one_of<std::ranges::range_value_t<T>, char, unsigned char, std::byte> && !
std::is_const_v<T>;

template <NonConstByteData Range>
int bulk_transfer(device_handle &dev, int endpoint, Range &&range,
                  std::chrono::milliseconds timeout) {
  using std::begin;
  using std::end;
  auto sent = int(0);
  auto err = libusb_bulk_transfer(
      dev.get(), endpoint, reinterpret_cast<unsigned char *>(&*begin(range)),
      ((end(range)) - (begin(range))), &sent, timeout.count());
  check(err);
  return sent;
}
