
int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
  auto device{Device(select_device_with_most_flops())};
}
