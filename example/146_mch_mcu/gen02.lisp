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

     ;; https://github.com/openwch/ch592/tree/main/EVT/EXAM/BLE/RF_PHY/APP
     ;; try rf communication module in basic mode
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
     (include<> cstdio)
     #+nil (include<> stdio.h
					;format
					;unistd.h
					;vector deque chrono
					;cmath
     
		      )

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
     
     (defun RF_Init ()
       (let ((cfg (rfConfig_t)))
	 (tmos_memset &cfg 0 (sizeof cfg))
	 (let ((task_id (uint8_t 0))))
	 (setf task_id (TMOS_ProcessEventRegister RF_ProcessEvent))
	 ,@(loop for (e f) in `((accessAddress (hex #x71764129))
				(CRCInit (hex #x555555))
				(Channel 39)
				(Frequency 2480000)
				(LLEMode (logior LLE_MODE_BASIC
						 LLE_MODE_EX_CHANNEL))
				(rfStatusCB RF_2G4StatusCallback)
				(RxMaxlen 251))
		 collect
		 `(setf (dot cfg ,e)
			,f))
	 (let ((state (RF_Config &cfg))))))
    
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
       #+nil
       (do0
	(comments "the blue led flashes. the BOOT button switches between a fast and a slow flashing frequency")
	(let ((tick (uint32_t 0))
	      (toggle_tick (uint32_t 250)))
	  (while 1
		 (incf tick)
		 (when (== 0 (% tick toggle_tick))
		   (board_led_toggle))
		 (when (board_button_getstate)
		   (while (board_button_getstate)
			  (DelayMs 50))
		   (if (== 250 toggle_tick)
		       (setf toggle_tick 100)
		       (setf toggle_tick 250)))
		 (DelayMs 1))))

       
       

       #+nil
       (do0
	(comments "For Debugging")
	(GPIOA_SetBits bTXD1)
	(GPIOA_ModeCfg bTXD1 GPIO_ModeOut_PP_5mA)
	(UART1_DefInit))
       #+nil

       (do0
	(PRINT (string "%s\\n") VER_LIB)
	(CH59x_BLEInit)
	(HalKeyConfig key_callback)
	(GAPRole_BroadcasterInit)
	(Broadcaster_Init)
	(Main_Circulation)
	)
       (return 0)
       
       )
     )
   :omit-parens t
   :format t
   :tidy nil))



