# Documentation of BLE HAL

I used Gemini Advanced 1.0 to summarize the header files in this folder.

## HAL.h

**HAL: Hardware Abstraction Layer**

* This header file is likely the core part of a Hardware Abstraction Layer (HAL) for a microcontroller from Nanjing Qinheng Microelectronics.
* **Purpose:** The HAL's goal is to provide a simplified, hardware-independent interface for programmers to interact with the microcontroller's various peripherals and features.

**Key Details**

* **Copyright:** Includes copyright notices (2016 and 2021) from potential authors/maintainers.
* **Disclaimer:** Emphasizes that the code is intended for Nanjing Qinheng Microelectronics manufactured devices.
* **Header Guards:** Employs standard `#ifndef __HAL_H...#define __HAL_H...#endif` to prevent multiple inclusions.
* **C++ Compatibility:** Includes `extern "C" { ... }` for compatibility with C++ projects.

**Includes:**

* **CONFIG.h:**  A  header file likely containing system-wide configurations and macro definitions.
* **RTC.h, SLEEP.h, LED.h, KEY.h:** Other HAL headers responsible for Real-Time Clock, sleep modes, LEDs, and keypad input, respectively. 

**Task Management:**

* **Events (e.g., LED_BLINK_EVENT):** Defines identifiers for different events or tasks the HAL might handle. 
* **tmosTaskID halTaskID:** Suggests that this HAL might integrate with a real-time operating system (likely TMOS) with 'halTaskID' being the task identifier for the HAL within that OS.

**Functions:**

* **HAL_Init():** Initializes the HAL, probably configuring all the included peripherals (RTC, sleep, LEDs, keypad).
* **HAL_ProcessEvent(tmosTaskID task_id, tmosEvents events):**  Seems to be the core event handling function within the HAL and may be linked to an OS. It processes a range of events signaled by the `events` bitmask. 
* **CH59x_BLEInit():** Specifically initializes Bluetooth Low Energy (BLE) functionality.
* **HAL_GetInterTempValue():**  Gets an internal temperature reading, likely using an onboard ADC (Analog-to-Digital Converter).
* **Lib_Calibration_LSI():** Calibrates the internal low-speed oscillator (LSI) of the microcontroller.

**Overall Impression**

This HAL seems to manage diverse low-level functionalities: timekeeping, power modes, input/output (LEDs, keypad), wireless communication (BLE), and even sensor readings. It presents a more user-friendly programming interface and hides hardware specifics by providing functions like `HAL_GetInterTempValue()`.



## SLEEP.h control low-power functionality

**Purpose:**

* This header file (`SLEEP.h`) provides code to interact with sleep/low-power functionality for a microcontroller likely manufactured by Nanjing Qinheng Microelectronics.
* It specifically targets power management features such as setting sleep modes and enabling wake-up functionality through an RTC (Real-Time Clock).

**Key Details**

* **Copyright:** Includes copyright notices (2018 and 2021) from potential authors / maintainers.
* **Disclaimer:** Emphasizes that the code is intended for Nanjing Qinheng Microelectronics manufactured devices.
* **Header Guards:** Employs standard `#ifndef __SLEEP_H...#define __SLEEP_H...#endif` to prevent multiple inclusions.
* **C++ Compatibility:** Includes `extern "C" { ... }` for compatibility with C++ projects.

**Functions**

* **HAL_SleepInit():** Initializes settings for sleep mode, likely configuring the RTC for wake-up events.
* **CH59x_LowPower(uint32_t time):**  Transitions the microcontroller into a low-power sleep state. The `time` argument probably specifies the sleep duration using RTC counter values.

**Important Notes**

* **Specific Hardware:**  Since the file mentions 'CH59x', it's likely that the code is tailored for a specific microcontroller family from Nanjing Qinheng Microelectronics.
* **RTC Dependency:** This header seems to rely on the microcontroller having a Real-Time Clock (RTC) to manage sleep-related timing and wake-up.

## RTC.h to manage Real-Time Clock

**Purpose**

* Provides code to manage a Real-Time Clock (RTC) on a microcontroller, likely from Nanjing Qinheng Microelectronics.
* Handles timekeeping, adjustments, and potentially RTC-based interrupts/alarms.

**Key Details**

* **Copyright:** Includes copyright notices (2016 and 2021) from potential authors/maintainers.
* **Disclaimer:** Emphasizes that the code is intended for Nanjing Qinheng Microelectronics manufactured devices.
* **Header Guards:** Employs standard `#ifndef __RTC_H...#define __RTC_H...#endif` to prevent multiple inclusions.
* **C++ Compatibility:** Includes `extern "C" { ... }` for compatibility with C++ projects.

**Constants:**

* **FREQ_RTC:** Determines the frequency of the RTC's oscillator (either 32000 Hz or 32768 Hz depending on the configuration).
* **CLK_PER_US / CLK_PER_MS:**  Clock cycles per microsecond/millisecond based on RTC frequency.
* **US_PER_CLK / MS_PER_CLK:** Microseconds/milliseconds per single RTC clock cycle.

**Macros:**

* **RTC_TO_US(clk) / RTC_TO_MS(clk):** Converts RTC clock values to microseconds/milliseconds.
* **US_TO_RTC(us) / MS_TO_RTC(ms):**  Converts microseconds/milliseconds to RTC clock values.

**Variables:**

* **RTCTigFlag:** A volatile flag likely used to signal RTC related events (e.g.,  time updates or alarms).

**Functions:**

* **HAL_TimeInit()**: Initializes the RTC and related timekeeping services.
* **RTC_SetTignTime(uint32_t time):** Sets the current RTC time (probably using an RTC counter value representation).

**Important Notes**

* **Oscillator Dependency:** The code expects a specific clock source (either 32000 Hz or 32768 Hz, likely a crystal oscillator) to be configured and connected to the RTC module.
* **Time Representation:** The RTC likely works with internal counter values; the provided macros help convert between these counter values and standard time units.


## KEY.h interface for physical buttons

**Purpose:**

* This header file provides an interface for interacting with physical keys/buttons on a microcontroller board (likely one manufactured by Nanjing Qinheng Microelectronics). 
* It aims to simplify key input handling by dealing with the low-level hardware details and offering more abstract functions.

**Key Details**

* **Copyright:** Includes copyright notices (2016 and 2021) from potential authors/maintainers.
* **Disclaimer:** Emphasizes that the code is intended for Nanjing Qinheng Microelectronics-manufactured devices.
* **Header Guards:** Employs standard `#ifndef __KEY_H...#define __KEY_H...#endif` to prevent multiple inclusions.
* **C++ Compatibility:** Includes `extern "C" { ... }` for compatibility with C++ projects.

**Macros**

* **HAL_KEY_POLLING_VALUE:** Likely defines an interval (in milliseconds) for checking the status of keys.
* **HAL_KEY_SW_1, HAL_KEY_SW_2, etc.:**  Represent individual keys as bit values (useful for bitwise operations).
* **KEY1_BV, KEY2_BV, etc.:** Bit values associated with physical microcontroller pins where keys are connected.
* **KEY1_PU, KEY2_PU, etc.:** Seem to configure internal pull-up resistors for the associated key input pins.
* **KEY1_DIR, KEY2_DIR, etc.:** Set the direction (input) for the key pins. 
* **KEY1_IN, KEY2_IN, etc.:** Read the input state of the key pins (ACTIVE_LOW likely implies keys pull the pin low when pressed).
* **HAL_PUSH_BUTTON1(), HAL_PUSH_BUTTON2() etc.:** Convenient functions to check the 'pressed' state of individual keys.

**Callback Mechanism**

* **halKeyCBack_t:** Defines a function pointer type for keypress callback functions.
* **keyChange_t:** A structure to hold information about which keys have changed state.

**Functions**

* **HAL_KeyInit():** Initializes the key input system (configures GPIO pins)
* **HAL_KeyPoll():**  Likely an internal function periodically checking key states.
* **HalKeyConfig(const halKeyCBack_t cback):**  Sets a callback function to be executed when a key event (press/release) occurs.  
* **HalKeyCallback(uint8_t keys):** Seems to be the actual callback invoked by the underlying framework, providing which keys changed state. 
* **HalKeyRead():** Returns the current state of the keys.

**Overall** 

This header offers functions to  initialize keys, get their states, and a  callback mechanism for asynchronous notification of key changes. It hides the details of interacting with low-level registers on the microcontroller.

## LED.h
