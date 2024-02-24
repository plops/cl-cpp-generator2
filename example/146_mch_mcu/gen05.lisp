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
  (defparameter *source-dir* #P"example/146_mch_mcu/source05/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  
  (let* ((name `Ch592UsbRegisters)
	 (l-regs `((:name ctrl :addr #x40008000
		    :fields ((:fname host-mode :bit 7 :access rw)
			     (:fname low-speed :bit 6 :access rw)
			     (:fname dev-pull-up-en :bit 5 :access rw)
			     (:fname sys-ctlr :bit (5 4) :access rw :help "host-mode==0: 00..disable usb device function and disable internal pull-up (can be overridden by dev-pullup-en), 01..enable device fucntion, disable internal pull-up, external pull-up-needed, 1x..enable usb device fucntion and internal 1.5k pull-up, pull-up has priority over pull-down resistor")
			     (:fname int-busy :bit 3 :access rw :help "Auto pause")
			     (:fname reset-sie :bit 2 :access rw :help "Software reset USB protocol processor")
			     (:fname clr-all :bit 1 :access rw :help "USB FIFO and interrupt flag clear")
			     (:fname dma-en :bit 0 :access rw)
			     
			     ))
		   (:name int-en :addr #x40008002
		    :fields ((:fname dev-sof :bit 7 :access rw :help "in device mode receive start of frame (SOF) packet interrupt")
			     (:fname dev-nak :bit 6 :access rw :help "in device mode receive NAK interrupt")
			     (:fname mod-1-wire-en :bit 5  :access rw :help "USB single line mode enable")
			     (:fname fifo-overflow :bit 4 :access rw :help "Fifo overflow interrupt")
			     (:fname host-sof :bit 3  :access rw :help "host start of frame timing interrupt")
			     (:fname suspend :bit 2 :access rw :help "USB bus suspend or wake-up event interrupt")
			     (:fname transfer :bit 1  :access rw :help "USB transfer completion interrupt")
			     (:fname bus-reset :bit 0  :access rw :help "in USB device mode USB bus reset event interrupt")
			     ))
		   (:name ctrl :addr #x40008000
		    :fields ((:fname host-mode :bit 7 :access rw :help "")
			     ))
		   ))
	 (members `((max-cores :type int :param t)
		    (max-points :type int :param t)
		    (diagrams :type "std::vector<DiagramData>")
					;(x :type "std::vector<float>")
					;(y :type "std::vector<float>")
		    (name-y :type "std::string" :param t)
		    (time-points :type "std::deque<float>"))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector deque string)
			(space struct DiagramData (progn
					   "std::string name;"
					   "std::deque<float> values;"
					   ))
			(doc "@brief The DiagramBase class represents a base class for diagrams."))
     :implementation-preamble
     `(do0
       
       (include<>
		  stdexcept
		  format
		  )
       )
     :code `(do0
	     
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname)) 
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor)
		  )
		 )

	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (get (cl-change-case:pascal-case (format nil "get-~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(defmethod ,get ()
					 (declare (values ,type))
					 (return ,nname_))))))
	       
	       "protected:"
	       
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_))))))))

    )

  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames (format nil "~a.cpp" module-name)
		     *source-dir*))
   
   `(do0
  

     (comments "based on https://github.com/openwch/ch592/tree/main/EVT/EXAM/USB/Device/VendorDefinedDev/src")
     (comments "try to write send data via USB to computer")
     (comments "AI summary of the example code is here: https://github.com/plops/cl-cpp-generator2/tree/master/example/146_mch_mcu/doc/examples/usb/device")

     (include<> 
      array
      cassert)
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

     (space constexpr uint16_t (= DevEP0Size (hex #x40)))
     (static_assert (< DevEP0Size 256)
		    (string "DevEP0Size must fit into one byte."))
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
	  (cond
	    ((& intflag RB_UIF_TRANSFER)
	     (do0
	      (unless (== MASK_UIS_TOKEN
			  (& R8_USB_INT_ST MASK_UIS_TOKEN))
		(do0
		 (comments "The following switch extracts the type of token (in/out) and the endpoint number.")
		 (case (& R8_USB_INT_ST (or MASK_UIS_TOKEN
					    MASK_UIS_ENDP))
		   (UIS_TOKEN_IN
		    (case SetupReqCode
		      (USB_GET_DESCRIPTOR
		       (comments "Handles the standard 'Get Descriptor' request. The device sends the appropriate descriptor data to the host.")
		       (comments "Calculate length of data to send. Limit to device endpoint size if needed.")
		       (let ((new_len (std--min DevEP0Size
						SetupReqLen)))
			 (assert (< new_len 256)))
		      
		       (setf len (static_cast<uint8_t> new_len)
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
		       (comments "Handles the standard 'Set Address' request. The device (we) records the new USB address.")
		       (assert (< SetupReqLen 256))
		       (setf R8_USB_DEV_AD (or (& R8_USB_DEV_AD RB_UDA_GP_BIT)
					       (static_cast<uint8_t> SetupReqLen)))
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
		   ((or UIS_TOKEN_OUT 1)
		    (comments "Handle data reception on endpoint 1.")
		    (when (& R8_USB_INT_ST
			     RB_UIS_TOG_OK)
		      (comments "If the data toggle is correct and data is ready.")
		      (comments "Toggles the receive (IN) data toggle bit for endpoint 1.")
		      (setf R8_UEP1_CTRL  (^  R8_UEP1_CTRL RB_UEP_R_TOG)
			    len R8_USB_RX_LEN)
		      (comments "Get the data length, and call a function (DevEP1_OUT_Deal) to process the received data (on endpoint 1).")
		      (DevEP1_OUT_Deal len)))

		   ((or UIS_TOKEN_IN 1)
		    (comments "Prepare an empty (?) response on endpoint 1.")
		    (comments "Toggle the transmit (OUT) data toggle bit for endpoint 1.")
		    (setf R8_UEP1_CTRL
			  (^ R8_UEP1_CTRL
			     RB_UEP_T_TOG))
		    (comments "Prepares endpoint 1 for a NAK response (indicating no data is ready to send).")
		    (setf R8_UEP1_CTRL
			  (or (& R8_UEP1_CTRL
				 ~MASK_UEP_T_RES)
			      UEP_T_RES_NAK)))
		   (t ) 
		   ))
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
					     len (dot ,f (at 0)))))
			     (t (comments "Unsupported string descriptor type.")
			      (setf errflag (hex #xff)))))
			  (t (setf errflag (hex #xff)))
			  )
			
			(comments "Limit the actual data sent based on the requested length.")
			(setf SetupReqLen (std--min SetupReqLen (static_cast<uint16_t> len)))
			(let ((new_len (std--min DevEP0Size SetupReqLen))))
			(assert (< new_len 256))
			(setf len (static_cast<uint8_t> new_len))
			(memcpy pEP0_DataBuf pDescr len)
			(incf pDescr len))
		       (USB_SET_ADDRESS
			(setf SetupReqLen (& pSetupReqPak->wValue (hex #xff))))
		       (USB_GET_CONFIGURATION
			(comments "Handles the 'Get Configuration' request (responds with the current device configuration).")
			(comments "Store configuration in the endpoint buffer for transmission.")
			(setf (aref pEP0_DataBuf 0) DevConfig)
			(comments "Ensure only a single byte is sent (as configuration is one byte).")
			(setf SetupReqLen (std--min (static_cast<uint16_t> 1) SetupReqLen)))
		       (USB_SET_CONFIGURATION
			(comments "Update the DevConfig variable with the new configuration value provided by the host.")
			(setf DevConfig (static_cast<uint8_t> (& pSetupReqPak->wValue (hex #xff)))))
		       (USB_CLEAR_FEATURE
			(comments "Clear endpoint stalls or other features.")
			(if (== USB_REQ_RECIP_ENDP
				(& pSetupReqPak->bRequestType
				   USB_REQ_RECIP_MASK))
			    (do0
			     (comments "Request targets an endpoint")
			     (comments "Clear stall conditions on specific enpoints (number in wIndex).")
			     (case (& pSetupReqPak->wIndex (hex #xff))
			       ((hex #x82)
				(setf R8_UEP2_CTRL (or (& R8_UEP2_CTRL
							  (~ (or RB_UEP_T_TOG
								 MASK_UEP_T_RES)))
						       UEP_T_RES_NAK)))
			       ((hex #x02)
				(setf R8_UEP2_CTRL (or (& R8_UEP2_CTRL
							  (~ (or RB_UEP_R_TOG
								 MASK_UEP_R_RES)))
						       UEP_R_RES_ACK)))
			       ((hex #x81)
				(setf R8_UEP1_CTRL (or (& R8_UEP1_CTRL
							  (~ (or RB_UEP_T_TOG
								 MASK_UEP_T_RES)))
						       UEP_T_RES_NAK)))
			       ((hex #x01)
				(setf R8_UEP1_CTRL (or (& R8_UEP1_CTRL
							  (~ (or RB_UEP_R_TOG
								 MASK_UEP_R_RES)))
						       UEP_R_RES_ACK)))
			       (t
				(comments "Unsupported endpoint number.")
				(setf errflag (hex #xff)))

			       )))
			)
		       (USB_GET_INTERFACE
			(comments "Retrieve the alternate setting of the current interface. It seems this device likely only has a single setting (always responds with 0).")
			(setf (aref pEP0_DataBuf 0) 0
			      SetupReqLen (std--min (static_cast<uint16_t> 1) SetupReqLen))
			)
		       (USB_GET_STATUS
			(comments "Get device or endpoint status. This implementation only supports a basic status response (all zeros).")
			(setf (aref pEP0_DataBuf 0) 0
			      (aref pEP0_DataBuf 1) 0
			      SetupReqLen (std--min (static_cast<uint16_t> 2) SetupReqLen)))
		       (t
			(comments "Catch-all for unsupported request codes. Sets an error flag.")
			(setf errflag (hex #xff))))))

		(if (== (hex #xff)
			errflag)
		    (do0
		     (comments "If the previously set errflag is 0xff (signaling an unsupported request), this code forces a STALL condition on the control endpoint. This signals to the host that the device doesn't recognize the request.")
		     (setf R8_UEP0_CTRL (or RB_UEP_R_TOG
					    RB_UEP_T_TOG
					    UEP_R_RES_STALL
					    UEP_T_RES_STALL)))
		    (do0
		     (doc "Determines Transfer Direction: Checks chtype. If the 0x80 bit is set, the host expects data from the device (upload/IN direction), otherwise, the host is sending data (download/OUT direction)."
			  "Sets the data transfer 2length (len) for this stage of the control transfer."
			  
			  )
		     (if (& (hex #x80)
			    chtype)
			 (do0
			  (comments "Upload")
			  (let ((new_len (std--min DevEP0Size SetupReqLen))))
			  (assert (< new_len 256))
			  (setf len (static_cast<uint8_t> new_len))
			  (decf SetupReqLen len))
			 (do0
			  (comments "Download")
			  (setf len 0)))
		     (comments "Configures Endpoint: Prepares the control endpoint register (R8_UEP0_CTRL) for data transmission (likely transitioning to the DATA1 stage of the control transfer).")
		     (setf R8_UEP0_T_LEN len
			   R8_UEP0_CTRL (or RB_UEP_R_TOG
					    RB_UEP_T_TOG
					    UEP_R_RES_ACK
					    UEP_T_RES_ACK)))
		    )
		(comments "Signals Completion: Sets an interrupt flag (R8_USB_INT_FG = RB_UIF_TRANSFER;) to indicate the setup process is finished.")
		(setf R8_USB_INT_FG RB_UIF_TRANSFER)
		)))
	    ((& intflag RB_UIF_BUS_RST)
	     (comments "A bus reset interrupt flag is detected...")
	     (doc "1. Reset Address: Clears the device's address (R8_USB_DEV_AD = 0;), putting it back to the default address state.
2. Reset Endpoints: Prepares all endpoints (endpoint 0 through 3) for new transactions.
3. Clear Interrupt Flag: Acknowledges the bus reset interrupt.")
	     (setf R8_USB_DEV_AD 0
		   R8_UEP0_CTRL (or UEP_R_RES_ACK UEP_T_RES_NAK)
		   R8_UEP1_CTRL (or UEP_R_RES_ACK UEP_T_RES_NAK)
		   R8_UEP2_CTRL (or UEP_R_RES_ACK UEP_T_RES_NAK)
		   R8_UEP3_CTRL (or UEP_R_RES_ACK UEP_T_RES_NAK)
		   R8_USB_INT_FG RB_UIF_BUS_RST))
	    ((& intflag RB_UIF_SUSPEND)
	     (comments "A suspend interrupt flag is detected...")
	     (doc "1. Check Suspend State: Reads a status register (R8_USB_MIS_ST & RB_UMS_SUSPEND) to determine if the device is truly suspended.
2. Suspend/Wake-up Actions: The commented sections could contain code to handle entering a low-power sleep mode (if suspended) or performing wake-up actions (if resuming from suspend).
3. Clear Interrupt Flag: Acknowledges the suspend interrupt.")
	     (if (& RB_UMS_SUSPEND R8_USB_MIS_ST)
		 (do0
		  (comments "Sleep"))
		 (do0
		  (comments "Wake up.")))
	     (setf R8_USB_INT_FG RB_UIF_SUSPEND)
	     )
	    (t
	     (comments "Catch any other unhandled interrupt flags and simply clears them.")
	     (setf R8_USB_INT_FG intflag))
	    ))))

     #+nil
     (defun DebugInit ()
       (doc "
  Sets a bit on GPIOA, Pin 9. This likely turns on an LED or some indicator 

  Configures GPIOA, Pin 8 as an input with internal pull-up resistor 

  
  Configures GPIOA, Pin 9 as a push-pull output with 5mA drive strength 

  
  Initializes UART1 with default settings. This sets up a serial port for debugging communication 

")
       (GPIOA_SetBits  GPIO_Pin_9)
       (GPIOA_ModeCfg GPIO_Pin_8 GPIO_ModeIN_PU)
       (GPIOA_ModeCfg GPIO_Pin_9 GPIO_ModeOut_PP_5mA)
       (UART1_DefInit))

     (defun main ()
       (declare (values int))
       (SetSysClock CLK_SOURCE_PLL_60MHz)
       (setf pEP0_RAM_Addr (EP0_Databuf.data))
       (USB_DeviceInit)
       (comments "Enable the interrupt associated with the USB peripheral.")
       (PFIC_EnableIRQ USB_IRQn)
       (while 1
	      (comments "inifinite loop")))


     (do0
      (doc "

__INTERRUPT is defined with __attribute__((interrupt('WCH-Interrupt-fast'))). This likely indicates a specialized, 'fast' interrupt mechanism specific to your compiler or microcontroller (WCH).


The compiler attribute __attribute__((section('.highcode'))) will be assigned to the __HIGH_CODE macro. This attribute likely instructs the compiler to place functions or code blocks marked with __HIGH_CODE into a special memory section named '.highcode' (possibly a faster memory region).


")
      (space __INTERRUPT
	     __HIGH_CODE
	     (defun USB_IRQHandler ()
	       (comments "Handle interrupts coming from the USB Peripheral")
	       (USB_DevTransProcess))))

     (defun DevEP1_OUT_Deal (l)
       (declare (type uint8_t l))
       (doc "Endpoint 1 data reception

1. l Parameter: The argument l represents the length (in bytes) of the received data packet.

2. Data Inversion: The core of the function is a loop that iterates through each received byte:

      pEP1_IN_DataBuf[i] = ~pEP1_OUT_DataBuf[i]; : This line inverts each byte of data (~ is the bitwise NOT operator) and stores the result in pEP1_IN_DataBuf.
3. Response Preparation:  The function calls DevEP1_IN_Deal(l).  This other function is likely responsible for sending the modified data (now in pEP1_IN_DataBuf) back to the host.
")
       (dotimes (i l)
	 (setf (aref pEP1_IN_DataBuf i)
	       (~ (aref pEP1_OUT_DataBuf i))))
       (DevEP1_IN_Deal l))
     
     )
   :omit-parens t
   :format t
   :tidy nil))



