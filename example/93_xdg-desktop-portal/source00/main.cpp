#define DBUS_API_SUBJECT_TO_CHANGE
#include <iostream>
#include <spdlog/spdlog.h>

int main(int argc, char **argv) {
  (void)argv;
  spdlog::info("start  argc='{}'", argc);
}
