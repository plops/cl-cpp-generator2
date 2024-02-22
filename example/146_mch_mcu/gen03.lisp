(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "alexandria")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))



(let ((module-name "main"))
  (defparameter *source-dir* #P"example/146_mch_mcu/source03/src/")
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
  

     (comments "based on https://github.com/openwch/ch592/tree/main/EVT/EXAM/USB/Device/VendorDefinedDev/src")
     (comments "try to write send data via USB to computer")
     (comments "AI summary of the example code is here: https://github.com/plops/cl-cpp-generator2/tree/master/example/146_mch_mcu/doc/examples/usb/device")
     (space extern "\"C\""
	    (progn
	      
	      (include<> 
	       CH59x_common.h)))
     (include<> 
		array)
     #+nil (include<> stdio.h
					;format
					;unistd.h
					;vector deque chrono
					;cmath
     
		      )

     (space const uint8_t DevEP0Size (hex #x40))
      (comments "vendor id and product id:")
      ,(let ((l `(#x12 1 #x10 1 #xff #x80 #x55 DevEP0Size #x48 #x43 #x37 #x55
		       0 1 1 2 0 1)))
	 `(space ,(format nil "std::array<uint8_t, ~a>" (length l))
		 DevDescr (curly
			   ,@(loop for e in l
				   collect
				   (if (numberp e)
				       (if (<= e 9)
					   e
					   `(hex ,e))
				       e)))))
      ,(let ((l `(9 2 #x4a 0 1 1 0 #x80 #x32 9 4 0 0 8 #xff
		    #x80 #x55 0 7 5 #x84 2 #x40 0 0 7 5 4 2 #x40
		    0 0 7 5 #x83 2 #x40 0 0 7 5 3 2 #x40 0
		    0 7 5 #x82 2 #x40 0 0 7 5 2 2 #x40 0 0
		    7 5 #x81 2 #x40 0 0 7 5 1 2 #x40 0 0)))
	 `(space ,(format nil "std::array<uint8_t, ~a>" (length l))
		 CfgDescr (curly
			   ,@(loop for e in l
				   collect
				   (if (numberp e)
				       (if (<= e 9)
					   e
					   `(hex ,e))
				       e)))))

      ,(let ((l `(4 3 9 4)))
	 `(space ,(format nil "std::array<uint8_t, ~a>" (length l))
		 ;; language descriptor
		 LangDescr (curly
			    ,@(loop for e in l
				    collect
				    (if (numberp e)
					(if (<= e 9)
					    e
					    `(hex ,e))
					e)))))

      ;; manufacturer information
      ,(let ((l `(#xe 3 (char "w") 0 (char "c") 0 (char "h") 0 (char ".") 0 (char "c") 0 (char "n"))))
	 `(space ,(format nil "std::array<uint8_t, ~a>" (length l))
		 ManuInfo (curly
			   ,@(loop for e in l
				   collect
				   (if (numberp e)
				       (if (<= e 9)
					   e
					   `(hex ,e))
				       e)))))

      ,(let ((l `(#xc 3
		      (char "C") 0
		      (char "H") 0
		      (char "5") 0
		      (char "9") 0
		      (char "x") 0)))
	 `(space ,(format nil "std::array<uint8_t, ~a>" (length l))
		 ProdInfo (curly
			   ,@(loop for e in l
				   collect
				   (if (numberp e)
				       (if (<= e 9)
					   e
					   `(hex ,e))
				       e)))))
      (space uint8_t DevConfig)
      (space uint8_t SetupReqCode)
      (space uint16_t SetupReqLen)
      (space const uint8_t* pDescr)

      ,@(loop for (e f) in `((EP0_Databuf ,(+ 64 64 64))
			     (EP1_Databuf ,(+ 64 64)))
	      collect
	      `(space (__attribute (paren (aligned 4)))
		      ,(format nil "std::array<uint8_t, ~a>" f)
		      ,e))


      (do0
       (doc "Handle USB transaction processing. Respond to standard USB requests (e.g. Get Descriptor, Set Address). Manage data transfers on endpoints.")
       (defun USB_DevTransProcess ()
	 (let ((len (uint8_t 0))
	       (chtype (uint8_t 0))
	       (errflag (uint8_t 0))))
	 (let ((intflag (uint8_t R8_USB_INT_FG)))
	   (when (& intflag RB_UIF_TRANSFER)
	     (unless (== MSK_UIS_TOKEN
			 (& R8_USB_INT_ST MASK_UIS_TOKEN))
	       (case (& R8_USB_INT_ST (or MASK_UIS_TOKEN
					  MASK_UIS_ENDP))
		 (UIS_TOKEN_IN
		  (case SetupReqCode
		    (USB_GET_DESCRIPTOR
		     (setf len (std--min DevEp0Size
					 SetupReqLen)
			   #+nil(? (<= DevEP0Size SetupReqLen)
			      DevEp0Size
			      SetupReqLen))
		     (memcpy ))))
		 ))))))
     
    )
   :omit-parens t
   :format t
   :tidy nil))



