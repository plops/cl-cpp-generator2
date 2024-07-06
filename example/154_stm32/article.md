
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

```
[ ] STM32G4-WDG_TIMERS-High_Resolution_Timer_HRTIM.pdf
[3] adc_internals_an2834_en.CD00211314.pdf
[ ] adc_oversampling_en.DM00722433.pdf
[1] adc_stm32g4_en.DM00625282.pdf
[ ] an3116-stm32s-adc-modes-and-their-applications-stmicroelectronics.pdf
[ ] an4539_dm00121475-hrtim-cookbook-stmicroelectronics.pdf
[2] analog_g4_en.DM00607955.pdf
[ ] stm32g4_hal.pdf
[ ] stm32g4_opamp_en.DM00605707.pdf

```

- v_in must have low impedance
  - on-chip opamps with 13MHz bandwidth can be used as followers (but
    i guess those wouldn't be of benefit for 5GHz signal)
  - how long does the charging of the internal sampling capacitor
    take? what is R_ADC [AN2834 p. 20],? do i need external Sample and
    hold circuit to sample with 5GHz?
  
- vref+ must have low impedance
- match adc dynamic range to input signal amplitude

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
