(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "alexandria"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))



(let ((module-name "main"))
  (defparameter *source-dir* #P"example/146_mch_mcu/source02/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
 

  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames (format nil "~a.cpp" module-name)
		     *source-dir*))
   
   `(do0
     #+nil(do0 
      (RF_Init)
      (RF_Config)
      (RF_Tx)
      (RF_Rx)
      (RF_GetStatus))

     (comments "based on https://github.com/openwch/ch592/tree/main/EVT/EXAM/BLE/RF_PHY/APP")
     (comments "try rf communication module in basic mode")
     (space extern "\"C\""
	    (progn
	      (include<> CONFIG.h
			 HAL.h
			; RF_PHY.h
			 )
	      (include<> 
	       CH59x_common.h
	       CH59x_sys.h
	       CH59x_pwr.h
	       board.h
					;HAL.h
					;broadcaster.h
	       )))
     (include<> cstdio
		array)
     #+nil (include<> stdio.h
					;format
					;unistd.h
					;vector deque chrono
					;cmath
     
		      )

     (space (__attribute (paren (aligned 4)))
	    uint32_t
	    (aref MEM_BUF (/ BLE_MEMHEAP_SIZE 4)))

     (space const uint8_t (aref MacAddr 6)
	     (curly (hex #x84)
		    (hex #xc2)
		    (hex #xe4)
		    3 2 2))

     (space "std::array<uint8_t, 10>" TX_DATA (curly 1 2 3 4 5 6 7 8 9 0))

     (space enum class SBP ": uint16_t" (curly START_DEVICE_EVT SBP_RF_PERIODIC_EVT SBP_RF_RF_RX_EVT))
     
     (space __HIGH_CODE
	    (__attribute (paren noinline))
	    (defun Main_Circulation ()
	      (while true
		     (TMOS_SystemProcess))))
     (doc
      "

**Purpose**

The `RF_2G4StatusCallBack` function acts as a central handler for status updates within a 2.4GHz radio frequency (RF) communication system. It's likely triggered by interrupts or events generated by the RF hardware. The function's key actions are:

* **Interpreting Status Codes:** It receives a status code (`sta`) that indicates the current state of the RF module (e.g., transmission finished, transmission failed, data received, timeout).
* **Updating Flags:**  It sets flags like `tx_end_flag` and `rx_end_flag` to control other parts of the program that are waiting for transmission or reception to complete. 
* **Data Handling:**  If data is received successfully (`crc` indicates no errors), it parses and potentially prints the received data to a console or debugging output.
* **Error Reporting:**  If errors occur (CRC check failures), it prints relevant error messages.
* **Event Triggering:** Depending on the configuration, it triggers events within the system  (`tmos_set_event`) to notify other tasks that data is ready or operations are completed.

**Translation of Comments**

* **@fn RF_2G4StatusCallBack** : Function name declaration
* **@brief RF status callback, this function is used within interrupts. Note: not to use this function with other APIs that handle receive or transmit, must work exclusively with the API used in interrupts. This callback or the APIs associated with it handle different conditions according to status.**
* **@param sta - Status type**
* **@param crc - CRC checksum result**
* **@param rxBuf - Data buffer pointer**
* **@return none**  

")
     (defun RF_2G4StatusCallback (sta crc rxBuf)
       (declare (type uint8_t sta crc)
		(type uint8_t* rxBuf))
       (case sta
	 (TX_MODE_TX_FINISH
	  )
	 (TX_MODE_TX_FAIL
	  )))

     (doc

      "
**Purpose**

The `RF_ProcessEvent` function acts as the event handler for a radio frequency (RF) module within a larger system. It processes different types of events that likely drive the overall RF communication operations.  Here's how it handles the various events:

* **`SYS_EVENT_MSG`:** 
   - This likely indicates a message from a task operating system (TMOS).
   - It processes the message and then releases the memory associated with it.

* **`SBP_RF_START_DEVICE_EVT`:**
   - This seems to be a device initialization event.
   - It starts a periodic task (`SBP_RF_PERIODIC_EVT`) with a 1000ms interval.

* **`SBP_RF_PERIODIC_EVT`:**
   - This is the core periodic event of the RF module.
   - It shuts down the RF module (`RF_Shut()`).
   - Clears the transmission end flag (`tx_end_flag = FALSE`).
   - Initiates a transmission (`RF_Tx()`) and waits for it to complete (`RF_Wait_Tx_End()`).
   - Reschedules itself to run again in 1000ms.

* **`SBP_RF_RF_RX_EVT`:**
   - This likely indicates the system wants to receive data.
   - Shuts down the RF module (`RF_Shut()`).
   - Prepares data for transmission (increments `TX_DATA[0]`).
   - Initiates data reception (`RF_Rx()`) and prints the state for debugging.

**Translation of Comments**

* **@fn RF_ProcessEvent** : Function name declaration
* **@brief RF event processing**
* **@param task_id - Task ID**
* **@param events - Event flags**
* **@return Returns unprocessed events** 

**Overall Behavior**

This function suggests a system design where:

* The RF module is periodically turned on to transmit data (likely sensor data or similar).
* The system can be triggered to enter a receive mode.


")
     
     (defun RF_ProcessEvent (task_id events)
       (declare (type uint8_t task_id)
		(type uint16_t events)
		(values uint16_t))
       (when (& events SYS_EVENT_MSG)))

     (doc "

RF_Wait_Tx_End() : This function waits for a transmission to end. It continuously checks a flag called tx_end_flag. If the flag isn't set (indicating the transmission is still ongoing), it enters a loop with a brief delay.  A timeout mechanism forces the tx_end_flag to TRUE after approximately 5ms.

RF_Wait_Rx_End() :  This function is very similar to the transmission wait function. It waits for a signal reception to end, monitoring the rx_end_flag. It also has a timeout mechanism of approximately 5ms.
")

     (doc
      "

**Purpose**

The `RF_Init` function initializes the radio frequency (RF) module, establishing its fundamental configuration.  Here's what it does step-by-step:

1. **Variable Setup:** Initializes  `state` for storing status and an  `rf_Config` structure for RF configuration.

2. **Task Registration:** Registers the `RF_ProcessEvent` function with the task operating system  (TMOS), enabling handling of RF-related events.

3. **RF Configuration:**
   * **`accessAddress`:**  Sets an access address (likely to identify packets for this device). The comment warns against using simple patterns like 0x55555555 or 0xAAAAAAAA.
   * **`CRCInit`:** Sets the initial value for CRC (Cyclic Redundancy Check) calculations used for error detection.
   * **`Channel`:** Sets the operating channel (39 in this case, representing a specific frequency within the 2.4 GHz range).
   * **`Frequency`:**  Fine-tunes the transmission frequency to 2480000 kHz (2.48 GHz).
   * **`LLEMode`:** Configures the Low-Level Link mode, which determines aspects of packet handling and potential auto-reply behaviors.  
   * **`rfStatusCB`:** Assigns the `RF_2G4StatusCallBack` function as the callback handler for status updates from the RF module.
   * **`RxMaxlen`:**  Sets the maximum length of data packets that can be received.

4. **RF Module Configuration:**  Calls the `RF_Config` function, passing the prepared `rf_Config` structure to apply the settings to the RF hardware.

5. **Status Print:** Prints the status returned by `RF_Config` (for initialization debugging purposes).

6. **Commented-Out Code:** There's a commented-out section that would have put the device into receive mode (`RF_Rx`) - this likely served testing purposes.

7. **Periodic Transmission Setup:**  Initializes periodic transmissions of data by triggering the `SBP_RF_PERIODIC_EVT`.

**Translation of Comments**

* **@fn RF_Init** : Function name declaration
* **@brief RF initialization** 
* **@return none**

**Key Notes**

* The specific RF chip used is not clear from the code, but the configuration parameters align with common 2.4GHz RF transceivers. 
* `RF_Auto_MODE_EXAM` seems to be a configuration flag; its exact behavior depends on how it's defined elsewhere in the system.



")
     
     (defun RF_Init ()
       (let ((cfg (rfConfig_t)))
	 (tmos_memset &cfg 0 (sizeof cfg))
	 (let ((taskID (uint8_t 0))))
	 (setf taskID (TMOS_ProcessEventRegister RF_ProcessEvent))
	 ,@(loop for (e f) in `((accessAddress (hex #x71764129))
				(CRCInit (hex #x555555))
				(Channel 39)
				(Frequency 2480000)
				(LLEMode (or LLE_MODE_BASIC
					     LLE_MODE_EX_CHANNEL))
				(rfStatusCB RF_2G4StatusCallback)
				(RxMaxlen 251))
		 collect
		 `(setf (dot cfg ,e)
			,f))
	 (let ((state (RF_Config &cfg))))

	 
	 (when false
	   (comments "RX mode")
	   (let ((state (RF_Rx (TX_DATA.data) (TX_DATA.size) (hex #xff) (hex #xff))))))

	 (when true
	   (comments "TX mode")
	   (tmos_set_event taskID (static_cast<uint16_t> SBP--SBP_RF_PERIODIC_EVT)))

	 

	 ))
    
     (defun main ()
       (declare (values int))
       #-nil
       (do0
	(comments "Enable DCDC")
	(PWR_DCDCCfg ENABLE))

       (SetSysClock CLK_SOURCE_PLL_60MHz)

       (do0
	(board_button_init)
	(board_led_init))
       (do0
	(comments "I think that fixes the gpio pins to prevent supply voltage from fluctuating")
	(GPIOA_ModeCfg GPIO_Pin_All GPIO_ModeIN_PU)
	(GPIOB_ModeCfg GPIO_Pin_All GPIO_ModeIN_PU))

       (CH59x_BLEInit)
       (HAL_Init)
       (RF_RoleInit)
       (RF_Init)
       (Main_Circulation)
       
       (return 0)
       
       )
     )
   :omit-parens t
   :format t
   :tidy nil))



