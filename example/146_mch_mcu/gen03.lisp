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

     (space constexpr uint8_t (= DevEP0Size (hex #x40)))
      (comments "vendor id and product id:")
      ,(let ((l `(#x12 1 #x10 1 #xff #x80 #x55 DevEP0Size #x48 #x43 #x37 #x55
		       0 1 1 2 0 1)))
	 `(space ,(format nil "const std::array<uint8_t, ~a>" (length l))
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
	 `(space ,(format nil "const std::array<uint8_t, ~a>" (length l))
		 CfgDescr (curly
			   ,@(loop for e in l
				   collect
				   (if (numberp e)
				       (if (<= e 9)
					   e
					   `(hex ,e))
				       e)))))

      ,(let ((l `(4 3 9 4)))
	 `(space ,(format nil "const std::array<uint8_t, ~a>" (length l))
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
	 `(space ,(format nil "const std::array<uint8_t, ~a>" (length l))
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
	 `(space ,(format nil "const std::array<uint8_t, ~a>" (length l))
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
	     (unless (== MASK_UIS_TOKEN
			 (& R8_USB_INT_ST MASK_UIS_TOKEN))
	       (case (& R8_USB_INT_ST (or MASK_UIS_TOKEN
					  MASK_UIS_ENDP))
		 (UIS_TOKEN_IN
		  (case SetupReqCode
		    (USB_GET_DESCRIPTOR
		     (comments "Handles the standard 'Get Descriptor' request. The device sends the appropriate descriptor data to the host.")
		     (comments "Calculate length of data to send. Limit to device endpoint size if needed.")
		     (setf len (std--min (static_cast<uint16_t> DevEP0Size)
					 SetupReqLen)
			   #+nil(? (<= DevEP0Size SetupReqLen)
				   DevEP0Size
				   SetupReqLen))
		     (comments "Copy the descriptor data to the endpoint buffer for transmission to the host.")
		     (memcpy pEP0_DataBuf pDescr len )
		     (decf SetupReqLen len)
		     (incf pDescr len)
		     (comments "Update state variables (length of the remaining request, pointer to the next chunk of descriptor data) and prepare for the next stage of the transfer.")
		     (setf R8_UEP0_T_LEN len
			   )
		     (setf R8_UEP0_CTRL (^ R8_UEP0_CTRL RB_UEP_T_TOG)))
		    (USB_SET_ADDRESS
		     (comments "Handles the standard 'Set Address' request. The device records the new USB address.")
		     (setf R8_USB_DEV_AD (or (& R8_USB_DEV_AD RB_UDA_GP_BIT)
					     SetupReqLen))
		     (setf R8_UEP0_CTRL (or UEP_R_RES_ACK UEP_T_RES_NAK)))
		    (t
		     (comments "Handles any other control requests. This usually results in a stall condition, as the device didn't recognize the request.")
		     (setf R8_UEP0_T_LEN 0
			   R8_UEP0_CTRL (or UEP_R_RES_ACK UEP_T_RES_NAK))))
		  )
		 (UIS_TOKEN_OUT
		  (comments "Handles 'OUT' token transactions, meaning the host is sending data to the device.")
		  (comments "Get length of received data.")
		  (setf len R8_USB_RX_LEN))
		 #+nil ((or UIS_TOKEN_OUT 1)
		  (when (& R8_USB_INT_ST
			   RB_UIS_TOG_OK)
		    (comments "f a particular status flag is set (indicating data is ready to be processed)")
		    (setf R8_UEP1_CTRL  (^  R8_UEP1_CTRL RB_UEP_R_TOG)
			  len R8_USB_RX_LEN)
		    (comments "Update state, get the data length, and call a function (DevEP1_OUT_Deal) to process the received data (on endpoint 1).")
		    (DevEP1_OUT_Deal len)))
		 (t )
		 )
	       (setf R8_USB_INT_FG RB_UIF_TRANSFER))
	     (comments "This code handles the initial 'Setup' stage of USB control transfers. When the host sends a setup packet to the device, this code analyzes the request and prepares a response.")
	     (when (& R8_USB_INT_ST
		      RB_UIS_SETUP_ACT)
	       (comments "A setup packet has been received.")
	       (comments "Prepare the control endpoint for a response.")
	       (setf R8_UEP0_CTRL (or RB_UEP_R_TOG
				      RB_UEP_T_TOG
				      UEP_R_RES_ACK
				      UEP_T_RES_NAK))
	       (comments "Extract the length, request code, and type from the setup packet.")
	       (setf SetupReqLen (-> pSetupReqPak wLength))
	       (setf SetupReqCode (-> pSetupReqPak bRequest))
	       (setf chtype (-> pSetupReqPak bRequestType))
	       (setf len 0
		     errflag 0)
	       (if (!= USB_REQ_TYP_STANDARD
			   (& chtype USB_REQ_TYP_MASK))
		   (do0 (comments "If the request type is NOT a standard request, set an error flag.")
			(setf errflag (hex #xff)))
		   (do0
		    (comments "Handle standard request.")
		    (case SetupReqCode
		      (USB_GET_DESCRIPTOR
		       (comments "Handle requests for device, configuration, or string descriptors.")
		       (case (>> pSetupReqPak->wValue 8)
			 (USB_DESCR_TYP_DEVICE
			  (setf pDescr (dot DevDescr (data))
				len (aref DevDescr 0)))
			 (USB_DESCR_TYP_CONFIG
			  (setf pDescr (dot CfgDescr (data))
				len (aref CfgDescr 2)))
			 (USB_DESCR_TYP_STRING
			  (case (& pSetupReqPak->wValue (hex #xff))
			    ,@(loop for (e f ) in `((0 LangDescr)
						    (1 ManuInfo)
						    (2 ProdInfo)
						    )
				    collect
				    `(,e
				     (setf pDescr (dot ,f (data))
					   len (aref (dot ,f (data)) 0))))
			    (t (comments "Unsupported string descriptor type.")
			     (setf errflag (hex #xff)))))
			 (t (setf errflag (hex #xff)))
			 ))
		     ))))))))
     
    )
   :omit-parens t
   :format t
   :tidy nil))



