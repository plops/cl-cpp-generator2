
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
