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
  (defparameter *source-dir* #P"example/146_mch_mcu/source01/src/")
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
     (space extern "\"C\""
	    (progn
	      
	      (include<> 
	       CH59x_common.h
	       HAL.h
	       broadcaster.h
	       )))
     (include<> cstdio)
     #+nil (include<> stdio.h
					;format
					;unistd.h
					;vector deque chrono
					;cmath
     
		      )
    (space
     __HIGH_CODE
     "__attribute__((noinline))"
     (defun Main_Circulation ()
       (while 1
	      (TMOS_SystemProcess))))

    (defun key_callback (keys)
      (declare (type uint8_t keys))
      (when (& keys HAL_KEY_SW_1)
	(printf (string "key pressed\\n"))
	(HalLedSet HAL_LED_ALL
		   HAL_LED_MODE_OFF)
	(HalLedBlink 1 2 30 1000)))
    
    (defun main ()
      (declare (values int))
      #+nil
      (do0
       (comments "Enable DCDC")
       (PWR_DCDCCfg ENABLE))
      (SetSysClock CLK_SOURCE_PLL_60MHz)

      #+nil
      (do0
       (comments "Enable Sleep.")
       (GPIOA_ModeCfg GPIO_Pin_All GPIO_ModeIN_PU)
       (GPIOB_ModeCfg GPIO_Pin_All GPIO_ModeIN_PU))

      #+nil
      (do0
       (comments "For Debugging")
       (GPIOA_SetBits bTXD1)
       (GPIOA_ModeCfg bTXD1 GPIO_ModeOut_PP_5mA)
       (UART1_DefInit))


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



