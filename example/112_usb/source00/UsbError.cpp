// no preamble
#include "UsbError.h"

void check(int err) {
  if (err < 0) {
    throw UsbError(err);
  }
}
UsbError::UsbError()
    : runtime_error(libusb_error_name(err_code)), _code(err_code) {}
int UsbError::code() const { return _code; }
