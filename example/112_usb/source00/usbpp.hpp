import fatheader;
template <typename T, void (*del)(T *)>
using Handle = std::unique_ptr<T, decltype([](T *x) { del(x); })>;
using context = Handle<libusb_context, libusb_exit>;

context init() {
  libusb_context *ctx = nullptr;
  check(libusb_init(&ctx));
  return context{ctx};
}
