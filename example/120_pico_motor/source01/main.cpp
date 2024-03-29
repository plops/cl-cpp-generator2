#include "motor_library.hpp"
#include <iostream>
enum { GPIO_OFF = 0, GPIO_ON = 1 };
enum { MOTOR1_IN1 = 2, LED_PIN = 25 };
volatile int state1 = 0;

void pio0_interrupt_handler() {
  pio_interrupt_clear(pio_0, 1);
  if (0 == state1) {
    SET_DIRECTION_MOTOR_1(COUNTERCLOCKWISE);
    state1 = 1;

  } else {
    if (1 == state1) {
      SET_DIRECTION_MOTOR_1(CLOCKWISE);
      state1 = 2;

    } else {
      SET_DIRECTION_MOTOR_1(STOPPED);
      state1 = 0;
    }
  }
  MOVE_STEPS_MOTOR_1(4048);
}

int main() {
  stdio_init_all();
  setupMotor1(MOTOR1_IN1, pio0_interrupt_handler);
  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

  pio0_interrupt_handler();
  while (true) {
    gpio_put(LED_PIN, GPIO_ON);
    sleep_ms(200);
    gpio_put(LED_PIN, GPIO_OFF);
    sleep_ms(200);
    std::cout << "hello world" << std::endl;
  }
  return 0;
}
