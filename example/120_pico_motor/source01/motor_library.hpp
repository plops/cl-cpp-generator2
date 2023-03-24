#include "counter.pio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/pio.h"
#include "pico/stdlib.h"
#include "stepper.pio.h"
#include <iostream>
enum { STOPPED = 0, CLOCKWISE = 1, COUNTERCLOCKWISE = 2 };
unsigned char pulse_sequence_forward[8] = {0b1001, 0b1000, 0b1100, 0b0100,
                                           0b0110, 0b0010, 0b0011, 0b0001};
unsigned char pulse_sequence_backward[8] = {0b0001, 0b0011, 0b0010, 0b0110,
                                            0b0100, 0b1100, 0b1000, 0b1001};
unsigned char pulse_sequence_stationary[8] = {0, 0, 0, 0, 0, 0, 0, 0};
unsigned int pulse_count_motor1 = 1024;
unsigned char *address_pointer_motor1 = pulse_sequence_forward;
unsigned int *pulse_count_motor1_address_pointer = &pulse_count_motor1;
#define MOVE_STEPS_MOTOR_1(a)                                                  \
  pulse_count_motor1 = a;                                                      \
  dma_channel_start(dma_chan_2)
#define SET_DIRECTION_MOTOR_1(a)                                               \
  address_pointer_motor1 = (a == 2)   ? pulse_sequence_forward                 \
                           : (a == 1) ? pulse_sequence_backward                \
                                      : pulse_sequence_stationary
PIO pio_0 = pio0;
int pulse_sm_0 = 0;
int count_sm_0 = 1;
// dma channels
// 0 .. pulse train to motor1
// 1 .. reconfigures and restarts irq 0
// 2 .. sends step count to motor 1
int dma_chan_0 = 0;
int dma_chan_1 = 1;
int dma_chan_2 = 2;

void setupMotor1(unsigned int in1, irq_handler_t handler) {
  auto pio0_offset_0 = pio_add_program(pio_0, &stepper_program);
  auto pio0_offset_1 = pio_add_program(pio_0, &counter_program);
  stepper_program_init(pio_0, pulse_sm_0, pio0_offset_0, in1);
  counter_program_init(pio_0, count_sm_0, pio0_offset_1);
  pio_sm_set_enabled(pio_0, pulse_sm_0, true);
  pio_sm_set_enabled(pio_0, count_sm_0, true);
  pio_interrupt_clear(pio_0, 1);
  pio_set_irq0_source_enabled(
      pio_0, static_cast<pio_interrupt_source>(PIO_INTR_SM1_LSB), true);
  irq_set_exclusive_handler(PIO0_IRQ_0, handler);
  irq_set_enabled(PIO0_IRQ_0, true);
  auto c0 = dma_channel_get_default_config(dma_chan_0);
  channel_config_set_transfer_data_size(&c0, DMA_SIZE_8);
  channel_config_set_read_increment(&c0, true);
  channel_config_set_write_increment(&c0, false);
  channel_config_set_dreq(&c0, DREQ_PIO0_TX0);
  channel_config_set_chain_to(&c0, dma_chan_1);
  dma_channel_configure(dma_chan_0, &c0, &(pio_0->txf[pulse_sm_0]),
                        address_pointer_motor1, 8, false);
  auto c1 = dma_channel_get_default_config(dma_chan_1);
  channel_config_set_transfer_data_size(&c1, DMA_SIZE_32);
  channel_config_set_read_increment(&c1, false);
  channel_config_set_write_increment(&c1, false);
  channel_config_set_chain_to(&c1, dma_chan_0);
  dma_channel_configure(dma_chan_1, &c1, &(dma_hw->ch[dma_chan_0].read_addr),
                        address_pointer_motor1, 1, false);
  auto c2 = dma_channel_get_default_config(dma_chan_2);
  channel_config_set_transfer_data_size(&c2, DMA_SIZE_32);
  channel_config_set_read_increment(&c2, false);
  channel_config_set_write_increment(&c2, false);
  dma_channel_configure(dma_chan_2, &c2, &(pio_0->txf[count_sm_0]),
                        pulse_count_motor1_address_pointer, 1, false);
  dma_start_channel_mask(1u << dma_chan_0);
}
