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
     (include 
	      /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/StdPeriphDriver/inc/CH59x_common.h
	      /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble/HAL/include/HAL.h
	      /home/martin/src/WeActStudio.WCH-BLE-Core/Examples/CH592/ble/broadcaster/ble/APP/include/broadcaster.h
	      )
    #+nil (include<> stdio.h
					;format
					;unistd.h
					;vector deque chrono
					;cmath
     
      )
     (defun main ()
       (do0
	(comments "Enable DCDC")
	(PWR_DCDCCfg ENABLE))
       (SetSysClock CLK_SOURCE_PLL_60MHz)

       (do0
	(comments "Enable Sleep.")
	(GPIOA_ModeCfg GPIO_Pin_All GPIO_ModeIN_PU)
	(GPIOB_ModeCfg GPIO_Pin_All GPIO_ModeIN_PU))

       (do0
	(comments "For Debugging")
	(GPIOA_SetBits bTXD1)
	(GPIOA_ModeCfg bTXD1 GPIO_ModeOut_PP_5mA)
	(UART1_DefInit))


       (do0
	(PRINT (string "%s\\n") VER_LIB)
	)
       
       )
     )
   :omit-parens t
   :format t
   :tidy nil))



