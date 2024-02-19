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
     
     (defun RF_2G4StatusCallback (sta crc rxBuf)
       (declare (type uint8_t sta crc)
		(type uint8_t* rxBuf))
       (case sta
	 (TX_MODE_TX_FINISH
	  )
	 (TX_MODE_TX_FAIL
	  )))
     
     (defun RF_ProcessEvent (task_id events)
       (declare (type uint8_t task_id)
		(type uint16_t events)
		(values uint16_t))
       (when (& events SYS_EVENT_MSG)))

     (doc "

RF_Wait_Tx_End() : This function waits for a transmission to end. It continuously checks a flag called tx_end_flag. If the flag isn't set (indicating the transmission is still ongoing), it enters a loop with a brief delay.  A timeout mechanism forces the tx_end_flag to TRUE after approximately 5ms.

RF_Wait_Rx_End() :  This function is very similar to the transmission wait function. It waits for a signal reception to end, monitoring the rx_end_flag. It also has a timeout mechanism of approximately 5ms.
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



