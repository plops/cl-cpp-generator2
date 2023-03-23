#include "counter.pio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/pio.h"
#include "pico/stdlib.h"
#include "stepper.pio.h"
#include <iostream>
enum { GPIO_OFF = 0, GPIO_ON = 1 };
enum { LED_PIN = 25 };

int main() {
  stdio_init_all();
  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

  while (true) {
    gpio_put(LED_PIN, GPIO_ON);
    sleep_ms(200);
    gpio_put(LED_PIN, GPIO_OFF);
    sleep_ms(200);

    std::cout << "hello world" << std::endl;
  }
  return 0;
}
