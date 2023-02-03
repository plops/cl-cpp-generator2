import fatheader;
template <typename T, void (*del)(T *)>
using Handle = std::unique_ptr<T, decltype([](T *x) { del(x); })>;
using context = Handle<libusb_context, libusb_exit>;

context init() {
  libusb_context *ctx = nullptr;
  check(libusb_init(&ctx));
  return context{ctx};
}

using device = Handle<libusb_device, libusb_unref_device>;

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

using device_handle = Handle<libusb_device_handle, libusb_close>;

device_handle open(device &dev) {
  libusb_device_handle *handle = nullptr;
  auto err = libusb_open(dev.get(), &handle);
  check(err);
  return device_handle{handle};
}

using device_descriptor = libusb_device_descriptor;

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
