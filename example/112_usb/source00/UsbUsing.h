#pragma once
#include <libusb-1.0/libusb.h>
#include <memory>
template <typename T, void (*del)(T *)>
using Handle = std::unique_ptr<T, decltype([](T *x) { del(x); })>;
using context = Handle<libusb_context, libusb_exit>;
using device = Handle<libusb_device, libusb_unref_device>;
using device_handle = Handle<libusb_device_handle, libusb_close>;
using device_descriptor = libusb_device_descriptor;
