#include "pico/stdlib.h"
#include <iostream>
enum { GPIO_OFF = 0, GPIO_ON = 1 };
enum { LED_PIN = 25 };

int main() {
  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

  setup_default_uart();
  while (true) {
    gpio_put(LED_PIN, GPIO_ON);
    sleep_ms(200);
    gpio_put(LED_PIN, GPIO_OFF);
    sleep_ms(200);

    std::cout << "hello world" << std::endl;
  }
  return 0;
}
