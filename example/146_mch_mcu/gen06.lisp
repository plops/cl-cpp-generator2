(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "alexandria")
  (ql:quickload "cl-change-case")
  (ql:quickload "str"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ((module-name "main"))
  (defparameter *source-dir* #P"example/146_mch_mcu/source06/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (load "util.lisp")
  (ensure-directories-exist *full-source-dir*)
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames (format nil "~a.cpp" module-name)
		     *source-dir*))
   
   `(do0
  

     (comments "based on https://github.com/openwch/ch592/blob/main/EVT/EXAM/UART1/src/Main.c")
     (comments "write debug message on UART")
     
     (include<> 
      array
      cassert

      )
     (space extern "\"C\""
	    (progn
	      
	      (include<> 
	       CH59x_common.h)))
     
     #+nil (include<> stdio.h
					;format
					;unistd.h
					;vector deque chrono
					;cmath
     
		      )
     

     (defun main ()
       (declare (values int))
       (SetSysClock CLK_SOURCE_PLL_60MHz)

       (GPIOA_SetBits GPIO_Pin_9)
       (GPIOA_ModeCfg GPIO_Pin_8 GPIO_ModeIN_PU)
       (GPIOA_ModeCfg GPIO_Pin_9 GPIO_ModeOut_PP_5mA)
       (comments  "This will configure UART to send and receive at 115200 baud:")
       (UART1_DefInit)

       ,(let ((msg-string "This s a tx test\\r\\n"))
	`(let ((TxBuf (,(format nil "std::array<uint8_t,~a>" (length msg-string))
			(string ,msg-string)))
	      
	       ;(trigB (uint8_t 0))
	       )))

       (when 1
	 (while true
	  (UART1_SendString (TxBuf.data) (TxBuf.size))))

       #+nil 
       (when 1
	 (let (
	       (RxBuf ("std::array<uint8_t,100>"))) 
	   (while true
		  (let ((len (UART1_RecvString (RxBuf.data)))))
		  (when len
		    (UART1_SendString (RxBuf.data)
				      len)))))
       
       )


     
     
     )
   :omit-parens t
   :format t
   :tidy nil))



