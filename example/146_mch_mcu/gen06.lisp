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
     ,(let ((msg-string "THis s a tx test\\r\\n"))
	`(let ((TxBuf (,(format nil "std::array<uint8_t,~a>" (length msg-string))
			(string ,msg-string)))
	       (RxBuf ("std::array<uint8_t,100>"))
	       (trigB (uint8_t 0)))))

     (defun main ()
       (declare (values int))
       (SetSysClock CLK_SOURCE_PLL_60MHz)

       (let ((len (uint8_t 0))))
       
       )


     
     
     )
   :omit-parens t
   :format t
   :tidy nil))



