
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



# Second Version

## II. STM32G4 ADC and HRTIM Capabilities

This section will delve into the specific capabilities of the STM32G4's ADC and HRTIM peripherals relevant to our objective of achieving high-speed equivalent-time sampling. We will also highlight key limitations that need to be addressed for successful implementation.

### 2.1 ADC Overview

The STM32G4 series features high-performance ADCs suitable for a variety of applications. Key specifications from [1, 2, 4, 10] relevant to equivalent-time sampling include:

* **Resolution:** The ADC offers programmable resolution, ranging from 6 to 12 bits. While 12-bit resolution is readily available, higher effective resolutions can be achieved using the built-in hardware oversampling feature [10], accumulating and averaging multiple samples. 
* **Sampling Rate:** The maximum ADC clock frequency is 60 MHz, translating to a maximum sampling rate of 4 Msps [1]. This poses a significant limitation for our 5.4GSps target, necessitating the use of equivalent-time sampling techniques.
* **Internal Channels:** The ADC provides access to a number of internal channels, including the temperature sensor, VBAT voltage, and internal reference voltage [1, 2]. These channels can be useful for calibration and system monitoring during equivalent-time sampling.
* **Triggers:**  The ADC can be triggered by both software and external events [1, 3], including timers and I/Os. The HRTIM peripheral plays a crucial role in generating precise, high-resolution triggers for equivalent-time sampling. 

### 2.2 HRTIM Overview

The HRTIM (High-Resolution Timer) peripheral is the cornerstone of our high-speed sampling strategy. Key features from [8, 9] relevant to our application include:

* **Clock:** The HRTIM can operate from a high-frequency clock source (up to 170 MHz) [8, 9], which is further divided by its internal Delay-Locked Loop (DLL) to achieve high resolution.
* **Resolution:** The DLL provides a 32-step division of the input clock period, enabling a time resolution as fine as 184 ps at 170 MHz [8, 9].  This resolution is crucial for precise ADC trigger placement across multiple signal cycles.
* **Timers:** The HRTIM consists of five identical timing units (Timers A to E) and a master timer [8, 9].  Each timing unit can generate two independent PWM outputs, offering a total of 10 outputs. The master timer facilitates synchronization and coordination between the timing units. 
* **Outputs:**  The HRTIM outputs can be configured in various modes, including single-shot, continuous, push-pull, and deadtime insertion [8, 9], offering flexibility in driving different power converter topologies.
* **Events:** The HRTIM provides a rich set of internal and external events [8, 9] for triggering actions such as output set/reset, counter reset, and capture events.  These events are essential for implementing complex waveform generation and precise ADC triggering schemes.
* **DMA:** The HRTIM supports DMA transfers for both output updates and event captures [9].  This offloads the CPU, enabling efficient data transfer and waveform generation.
* **ADC Triggers:** The HRTIM offers dedicated ADC trigger channels [8, 9] that can be synchronized with its timer events, allowing for precisely timed ADC conversions across multiple signal cycles.

### 2.3 Limitations

While the STM32G4's ADC and HRTIM offer a promising foundation for equivalent-time sampling, certain limitations need careful consideration:

* **Minimum ADC Sampling Time:** The minimum sampling time for the ADC is 2.5 ADC clock cycles, which translates to 42 ns at 60 MHz [4].  This duration might be insufficient for accurately capturing very high-speed signals, requiring signal conditioning or external sample-and-hold circuits.
* **External OpAmp Bandwidth:** When amplifying high-speed signals for shorter ADC sampling times, external operational amplifiers with bandwidth significantly exceeding the target sampling rate (5.4 GS/s) are necessary to minimize distortion [5]. 
* **Jitter:**  The DAC, which is used for generating the trigger for the ADC, captures the trigger signal from the APB clock. Even when timers can operate at higher frequencies, the DAC timing is limited by the APB clock. This could introduce jitter in the ADC trigger timing, degrading the effective sampling rate [6]. Estimates for this jitter need to be determined based on the specific APB clock frequency and its impact on the desired time resolution assessed.
* **VSSA on WeAct Core Board:**  The review uncovered a potential issue with the WeAct Core Board, where the VSSA pin is not directly connected to the chip's bonding pad [review]. This could lead to increased inductance and noise on the analog ground reference. Alternative low-impedance ground connections will need to be considered. 

By understanding these capabilities and limitations, we can develop effective strategies for implementing equivalent-time sampling on the STM32G4, as will be discussed in the next section.
