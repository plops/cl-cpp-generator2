// no preamble
#include "Error.h"
Error::Error() : runtime_error(libusb_error_name(err_code)), _code(err_code) {}
int Error::code() const { return _code; }
