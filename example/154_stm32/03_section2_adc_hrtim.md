
## II. STM32G4 ADC and HRTIM Capabilities

This section provides an overview of the relevant features and specifications of the STM32G4's ADC and HRTIM peripherals that are crucial for implementing equivalent-time sampling.

### 2.1 ADC Overview

The STM32G4 series integrates high-performance ADCs, featuring:

* **Resolution:** 12 bits (optionally configurable to 10, 8, or 6 bits). Additionally, the hardware oversampling feature can extend the resolution up to 16 bits by averaging multiple samples ([1, 2]).
* **Sampling Rate:** Up to 4 Msps (Mega Samples Per Second) when operating with a 60 MHz ADC clock ([1]).
* **Input Channels:** Up to 42 external channels (GPIOs) that can be configured for single-ended or differential measurements ([1, 2, 9]).
* **Internal Channels:** Access to internal signals such as temperature sensor, VBAT/3, VREFINT, and OPAMP outputs, providing valuable information about the chip's operating conditions ([2]).
* **Triggers:**  The ADC can be triggered by software or a wide range of external sources, including timers, I/Os, and the HRTIM ([1, 2, 9]). The STM32G4 offers up to 32 external trigger sources, enhancing flexibility for complex timing schemes.
* **Modes:**  Supports various conversion modes such as single, continuous, scan, discontinuous, and injected, offering flexibility for different sampling requirements ([1, 2]).

### 2.2 HRTIM Overview

The High-Resolution Timer (HRTIM) is a key peripheral for achieving precise timing control in power conversion applications. Notable features include:

* **Clock:**  Operates with a clock frequency ranging from 100 MHz to 170 MHz, derived from the PLL output ([8, 9]).
* **Resolution:** A Delay-Locked Loop (DLL) divides the timer's input clock period into 32 steps, effectively increasing the resolution.  This yields a timing resolution ranging from 312 ps down to 184 ps, depending on the input clock frequency ([8]).
* **Timers:**  Comprises seven independent 16-bit counters: six dedicated timing units (Timer A to F) and a master timer for synchronization ([8, 9]). Each timer has its own clock prescaler, allowing for different time bases.
* **Outputs:**  Capable of generating up to 12 output signals ([8, 9]). These outputs can be configured as PWM, complementary PWM with dead-time insertion, and arbitrary waveforms.
* **Events:**  Supports 10 external events and 6 fault signals, allowing for dynamic waveform modifications and fault protection ([8, 9]). These events can be triggered by external signals, internal comparators, or other timers.
* **DMA:**  Dedicated DMA channels allow for efficient data transfer between the HRTIM, memory, and other peripherals ([8, 9]). This offloads the CPU from repetitive tasks, enabling more complex control algorithms.
* **ADC Triggers:**  The HRTIM can generate precise triggers for ADC conversions ([8, 9]). This synchronization allows for sampling at specific points within the PWM cycle.

### 2.3 Limitations

While the STM32G4 ADC and HRTIM provide a powerful foundation for equivalent-time sampling, several limitations must be considered:

* **Minimum ADC Sampling Time:** The minimum ADC sampling time is 2.5 ADC clock cycles ([1, 4]), corresponding to 42 ns with a 60 MHz ADC clock. This limits the achievable time resolution, especially for high-frequency signals.
* **External OpAmp Bandwidth:**  For signals exceeding the ADC's native bandwidth, a high-speed external OpAmp is necessary for amplification and buffering ([4]). The choice of OpAmp must consider slew rate, gain-bandwidth product, and settling time.
* **APB Clock Jitter:**  While some timers can operate with frequencies exceeding the APB clock, the DAC captures the trigger signal only at the APB clock frequency ([7]). This can introduce jitter in the DAC output, potentially affecting the timing accuracy of ADC triggers.

These limitations will be addressed in later sections, presenting implementation strategies and practical solutions to mitigate their impact on the overall system performance.
