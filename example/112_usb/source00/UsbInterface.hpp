#pragma once
#include "UsbError.h"
#include "UsbUsing.h"
#include <libusb-1.0/libusb.h>
UsbInterface::UsbInterface(int i, device_handle &dev)
    : handle(i), dev(dev.get()) {}
