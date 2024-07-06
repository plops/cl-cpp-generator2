
```
[6] STM32G4-Analog-DAC.pdf
[ ] STM32G4-WDG_TIMERS-High_Resolution_Timer_HRTIM.pdf
[4] adc_internals_an2834_en.CD00211314.pdf
[ ] adc_oversampling_en.DM00722433.pdf
[1] adc_stm32g4_en.DM00625282.pdf
[3] an3116-stm32s-adc-modes-and-their-applications-stmicroelectronics.pdf
[ ] an4539_dm00121475-hrtim-cookbook-stmicroelectronics.pdf
[2] analog_g4_en.DM00607955.pdf
[ ] stm32g4_hal.pdf
[5] stm32g4_opamp_en.DM00605707.pdf

```

- the package contained in weact core board doesn't provide vssa 
https://github.com/WeActStudio/WeActStudio.STM32G474CoreBoard/blob/master/Hardware/WeAct-STM32G474CoreBoard_L_V10_SchDoc.pdf

- apparently when vssa is present this pin is connected to the bonding pad of the ic using double bonding wire to provide a low inductance path

- a 100nF and 1uF capacitor is connected to vref+ that seems to be okay.
- i can't figure out if these capacitors are close to the chip and
  connected to large copper planes, though
  
- an2834 explains a lot of non-idealities of the adc. perhaps i can
  attempt to measure them for my device (offset error, gain error,
  differential linearity error, integral linearity error)

- the pcb contains a linear regulator to convert 5v from the usb bus
  to 3.3v power supply. i guess for best measurement results i should
  supply a very clean supply voltage (not from switched mode
  powersupply).

# an3116 [3]

- don't run adc's at the same time (leave some dead time to prevent
  cross talk and too much load on vref+).

# an2834 [4]
- v_in must have low impedance
  - on-chip opamps with 13MHz bandwidth can be used as followers (but
    i guess those wouldn't be of benefit for 5GHz signal)
  - how long does the charging of the internal sampling capacitor
    take? what is R_ADC [AN2834 p. 20],? do i need external Sample and
    hold circuit to sample with 5GHz?
  
- vref+ must have low impedance
- match adc dynamic range to input signal amplitude
- vref+ should be created by a linear supply (not switching-type power supply)

- external vref+ must be at least 2.4V

- they suggest to add noise or triangular sweep to a perfectly
  constant V_in, so that ADC results can be averaged to higher
  precision~

- sampling duration increases with ADC resolutino AN2834 p. 32.
  - 17.1 ns for 8-bit
  - 67.1 ns for 16-bit (just the input sampling time to charge the
    sampling capacitor, assuming R_AIN=1kOhm)
  - when ADC resolution is 8-bit or lower and acquisition accuracy is
    allowed to be > +/- 1 LSB then T_SMPL can be minimized to a few
    picoseconds

- for STM32G4 the minimum sampling time is 2.5 ADC clock cycles. The
  maximum ADC clock frequency is 60MHz (i think). this corresponds to
  42ns
  
- a SAR ADC sampling estimation tool is available on demand

- if external amplifier is used then the ADC sampling time must be
  chosen to be several times longer than the amplifier propagation
  delay

- shield should only be grounded at the receiver (using analog ground)
- signal ground should come in a separate cable

- separate digital lines with ground tracks to minimize cross talk

- also separate crystals using ground tracks/ plane

- adc calibration by using a mathematical model of the adc implementation
  - offset
  - gain
  - bit weight

- use minimum disturbance from microcontroller during adc conversion
  - minimize i/o pin changes
  - CPU stop, wait mode
  - stop clock for unnecessary peripherals (timers, communication)

- for slow measurements on extra high impedance sources you can add a
  capacitor to the input pin that at least equals the internal
  capacitor (16pF) times U_max / U_lsb, e.g. 16pF * 4095 / 0.5 =
  131nF. (i'm not sure i understand this)
  - for some reason they say that the internal capacitor will charge
    the external one, strange. 
  - a larger external capacitor decreases the wait time between
    conversions but limits frqeuency bandwidth
  - the switch itself has parasitic capacitances. the pmos and nmos
    transistors have different capacities so that switching can charge
    sampling capacitor

# an5306 opamp [5]

- input offset +/- 3mV
- bandwidth 13MHz 
- slew rate 6.5 or 45 V/us (high speed uses more power and is typically used to amplify dac)
- gain -63, -31, .., -1, 1, 2, .. 64 (use external resistors or capacitors for other values or filters, e.g. as anti-alias filter)  
- gain error 2%
- open loop gain 95dB (56000x)
- wakup time 3us

- VINP3 can be internally connected to the dac outputs (use dac 3 or 4)  
- output can be routed to external pin or to ADC channel 

- offset calibration is documented in reference manual

- inverted configuration with offset works with few external components

- can i use the dac at 15MHz to mix the signal down into the bandwidth
  of the ADC? i guess not because it is not performing a
  multiplication.
  - maybe the timer controlled multiplexer mode can be used for
    that. how fast can the input be switched?

# Dac [6]

- 8 or 12 bit
- 1Msps (external) or 15Msps (internal, i think it can be routed out using OPAMP)
- buffered (only for external pin with slow dac) on non-buffered
- sample and hold (can hold static value in stop mode for low-power)
- two converters in one DAC module
- can generate noise
