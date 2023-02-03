template <typename T, void (*del)(T *)>
using Handle = std::unique_ptr<T, decltype([](T *x) { del(x); })>;
using context = Handle<libusb_context, libusb_exit>;

void check(int err) {
  if (err < 0) {
    throw Error(err);
  }
}

context init() {
  auto *ctx = libusb_context * (nullptr);
  check(libusb_init(&ctx));
  return context{ctx};
}
