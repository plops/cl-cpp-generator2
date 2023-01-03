using namespace cl;
int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  auto platforms = std::vector<Platform>();
  auto platformDevices = std::vector<Device>();
  try {
    Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
    auto context = Context(platformDevices);
    auto ctxDevices = context.getInfo<CL_CONTEXT_DEVICES>();
    for (auto &d : ctxDevices) {
      auto name = d.getInfo<CL_DEVICE_NAME>();
      spdlog::info("  name='{}'", name);
    }
  } catch (cl::Error const &e) {
    spdlog::info("  e.what()='{}'  e.err()='{}'", e.what(), e.err());
  };
}