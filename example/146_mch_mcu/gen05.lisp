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
  (defparameter *source-dir* #P"example/146_mch_mcu/source05/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (load "util.lisp")
  (ensure-directories-exist *full-source-dir*)
  ;; instead of manually transferring the register settings from the manual, i should parse this xml file: CH59Xxx.svd 
  ;; the xml contains all the required information (even read-only)
  ;; i can't find the fields of R32_USB_STATUS in the pdf or xml
  (let* ((name `Ch592UsbRegisters)
	 (ds-regs `((:name ctrl :ds-name R8_USB_CTRL :addr #x40008000) 
		    (:name port-ctrl :ds-name R8_UDEV_CTRL :addr #x40008001)
		    (:name int-en :ds-name R8_USB_INT_EN :addr #x40008002) 
		    (:name dev-ad :ds-name R8_USB_DEV_AD :addr #x40008003) 
		    (:name :ds-name R32_USB_STATUS :addr #x40008004) 
		    (:name misc-status :ds-name R8_USB_MIS_ST :addr #x40008005) 
		    (:name int-flag :ds-name R8_USB_INT_FG :addr #x40008006) 
		    (:name int-status :ds-name R8_USB_INT_ST :addr #x40008007) 
		    (:name rx-len :ds-name R8_USB_RX_LEN :addr #x40008008) 
		    (:name ep4-1-mod :ds-name R8_UEP4_1_MOD :addr #x4000800c) 
		    (:name ep2-3-mod :ds-name R8_UEP2_3_MOD :addr #x4000800d) 
		    (:name ep567-mod :ds-name R8_UEP567_MOD :addr #x4000800e) 
		    (:name ep0-dma :ds-name R16_UEP0_DMA :addr #x40008010) 
		    (:name ep1-dma :ds-name R16_UEP1_DMA :addr #x40008014) 
		    (:name ep2-dma :ds-name R16_UEP2_DMA :addr #x40008018) 
		    (:name ep3-dma :ds-name R16_UEP3_DMA :addr #x4000801c) 
		    (:name ep0-t-len :ds-name R8_UEP0_T_LEN :addr #x40008020) 
		    (:name ep0-ctrl :ds-name R8_UEP0_CTRL :addr #x40008022) 
		    (:name ep1-t-len :ds-name R8_UEP1_T_LEN :addr #x40008024) 
		    (:name ep1-ctrl :ds-name R8_UEP1_CTRL :addr #x40008026) 
		    (:name ep2-t-len :ds-name R8_UEP2_T_LEN :addr #x40008028) 
		    (:name ep2-ctrl :ds-name R8_UEP2_CTRL :addr #x4000802a) 
		    (:name ep3-t-len :ds-name R8_UEP3_T_LEN :addr #x4000802c) 
		    (:name ep3-ctrl :ds-name R8_UEP3_CTRL :addr #x4000802e) 
		    (:name ep4-t-len :ds-name R8_UEP4_T_LEN :addr #x40008030) 
		    (:name ep4-ctrl :ds-name R8_UEP4_CTRL :addr #x40008032) 
		    (:name ep5-dma :ds-name R16_UEP5_DMA :addr #x40008054) 
		    (:name ep6-dma :ds-name R16_UEP6_DMA :addr #x40008058) 
		    (:name ep7-dma :ds-name R16_UEP7_DMA :addr #x4000805c) 
		    (:name ep5-t-len :ds-name R8_UEP5_T_LEN :addr #x40008064) 
		    (:name ep5-ctrl :ds-name R8_UEP5_CTRL :addr #x40008066) 
		    (:name ep6-t-len :ds-name R8_UEP6_T_LEN :addr #x40008068) 
		    (:name ep6-ctrl :ds-name R8_UEP6_CTRL :addr #x4000806a) 
		    (:name ep7-t-len :ds-name R8_UEP7_T_LEN :addr #x4000806c) 
		    (:name ep7-ctrl :ds-name R8_UEP7_CTRL :addr #x4000806e) 
		    (:name epx-mode :ds-name R32_EPX_MODE :addr #x40008070)))
	 (l-regs `((:name ctrl :addr #x40008000
		    :fields ((:fname host-mode :bit 7 :access rw)
			     (:fname low-speed :bit 6 :access rw)
					;  (:fname dev-pull-up-en :bit 5 :access rw)
			     (:fname sys-ctlr :bit (5 4) :access rw :help "host-mode==0: 00..disable usb device function and disable internal pull-up (can be overridden by dev-pullup-en), 01..enable device fucntion, disable internal pull-up, external pull-up-needed, 1x..enable usb device fucntion and internal 1.5k pull-up, pull-up has priority over pull-down resistor")
			     (:fname int-busy :bit 3 :access rw :help "Auto pause")
			     (:fname reset-sie :bit 2 :access rw :help "Software reset USB protocol processor")
			     (:fname clr-all :bit 1 :access rw :help "USB FIFO and interrupt flag clear")
			     (:fname dma-en :bit 0 :access rw)
			     ))

		   (:name port-ctrl :addr #x40008001
		    :fields ((:fname pd-dis :bit 7 :access rw :help "disable USB-UDP-UDM pulldown resistance")
			     (:fname reserved6 :bit 6 :access ro)
			     (:fname dp-pin :bit 5 :access ro :help "UDP pin level")
			     (:fname dm-pin :bit 4 :access ro :help "UDM pin level")
			     (:fname reserved3 :bit 3 :access ro)

			     (:fname low-speed :bit 2 :access rw :help "enable USB port low speed (0==full speed, 1== low speed)")
			     (:fname hub0-reset :bit 1 :access rw :help "0=normal 1=force bus reset")
			     (:fname port-en :bit 0 :access rw :help "enable USB physical port (disabled automatically when device detached)")
			     
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
		   (:name dev-ad :addr #x40008003
					;:size 4
			  :fields ((:fname gp-bit :bit 7 :access rw :help "USB general flag, user-defined")
				   (:fname usb-addr :bit (6 0) :access rw :help "device mode: the address of the USB itself")
				   ))
		   #+nil (:name status :addr #x40008004
			  :fields ((:fname status :bit 7 :access rw :help "")
				   ))
		   (:name misc-status :addr #x40008005
		    :reg-access ro
		    :fields ((:fname sof-pre :bit 7 :access ro :help "SOF packet will be sent in host mode")
			     (:fname sof-act :bit 6 :access ro :help "SOF packet is being sent in host mode")
			     (:fname sie-free :bit 5 :access ro :help "USB proctocol processor free (not busy)")
			     (:fname r-fifo-rdy :bit 4 :access ro :help "USB receiver fifo data ready status (not empty)")
			     (:fname bus-reset :bit 3 :access ro :help "USB bus reset (is at reset status)")
			     (:fname bus-suspend :bit 2 :access ro :help "USB suspend status (is in suspended status)")
			     (:fname dm-level :bit 1 :access ro :help "In USB host mode, the level status of data minus (D-, DM) pin when the device is just connected to the USB port. used to determine speed (high level, = low speed)")
			     (:fname dev-attach :bit 0 :access ro :help "USB device connection status of the port in USB host mode (1 == port has been connected)")
			     ))

		   (:name int-flag  :addr #x40008006
		    :fields ((:fname is-nak :bit 7 :access ro :help "in device mode: NAK acknowledge during current USB transmission")
			     (:fname tog-ok :bit 6 :access ro :help "USB transmission data synchronous flag match status (1==synchronous, 0==asynchronous)")
			     (:fname sie-free :bit 5 :access ro :help "USB processor is idle")
			     (:fname fifo-ov :bit 4 :access rw :help "USB FIFO overflow interrupt flag. Write 1 to reset")
			     (:fname hst-sof :bit 3 :access rw :help "SOF packet transmission completion trigger in USB host mode. Write 1 to reset.")
			     (:fname suspend :bit 2 :access rw :help "USB suspend or wake-up event trigger. Write 1 to reset.")
			     (:fname transfer :bit 1 :access rw :help "USB transmission completion trigger. Write 1 to reset.")
			     (:fname bus-reset :bit 0 :access rw :help "in device mode: bus reset event trigger. Write 1 to reset.")
			     ))
		   (:name int-status  :addr #x40008007
		    :reg-access ro
		    :fields ((:fname setup-act :bit 7 :access ro :help "in device mode, when this bit is 1, 8-byte setup request packet has been successfully received.")
			     (:fname tog-ok :bit 6 :access ro :help "current usb transmission sync flag matching status (same as RB_U_TOG_OK), 1=>sync")
			     (:fname token :bit (5 4) :access ro :help "in device mode the token pid of the current usb transfer transaction")
			     (:fname endp :bit (3 0) :access ro :help "in device mode the endpoint number of the current usb transfer transaction")
			     ))
		   (:name rx-len :addr #x40008008
		    :reg-access ro
		    :fields ((:fname len :bit (6 0) :access ro :help "number of data bytes received by the current usb endpoint")
			     (:fname reserved7 :bit 7 :access ro )
			     ))

		   (:name reserved8009 :addr #x40008009
		    :reg-access ro)
		   (:name reserved800a :addr #x4000800a
		    :reg-access ro)
		   (:name reserved800b :addr #x4000800b
		    :reg-access ro)

		   (:name ep4-1-mod :addr #x4000800c
		    :reg-access rw
		    :fields (
			     (:fname ep1-rx-en :bit 0 :access rw :help "enable endpoint 1 receiving (OUT)")
			     (:fname ep1-tx-en :bit 1 :access rw :help "enable endpoint 1 transmittal (IN)")
			     (:fname reserved2 :bit 2 :access ro)
			     (:fname ep1-buf-mod :bit 3 :access rw :help "endpoint 1 buffer mode")
			     (:fname ep4-rx-en :bit 4 :access rw :help "enable endpoint 4 receiving (OUT)")
			     (:fname ep4-tx-en :bit 5 :access rw :help "enable endpoint 4 transmittal (IN)")
			     (:fname reserved76 :bit (7 6) :access ro )
			     ))

		   (:name ep2-3-mod :addr #x4000800d
		    :reg-access rw
		    :fields (
			     
			     (:fname ep3-rx-en :bit 0 :access rw )
			     (:fname ep3-tx-en :bit 1 :access rw )
			     (:fname reserved2 :bit 2 :access ro )
			     (:fname ep3-buf-mod :bit 3 :access rw )
			     (:fname ep2-rx-en :bit 4 :access rw)
			     (:fname ep2-tx-en :bit 5 :access rw)
			     (:fname reserved6 :bit 6 :access ro )
			     (:fname ep2-buf-mod :bit 7 :access rw)

			     ))
		   (:name ep567-mod :addr #x4000800e
		    :reg-access rw
		    :fields (
			     (:fname reserved01 :bit (1 0) :access ro )
			     (:fname ep7-rx-en :bit 2 :access rw )
			     (:fname ep7-tx-en :bit 3 :access rw)
			     (:fname ep6-rx-en :bit 4 :access rw )
			     (:fname ep6-tx-en :bit 5 :access rw)
			     (:fname ep5-rx-en :bit 6 :access rw )
			     (:fname ep5-tx-en :bit 7 :access rw)
			     ))
		   (:name reserved800f :addr #x4000800f
		    :reg-access ro)
		   ,@(loop for e in `(0 1 2 3)
			   appending
			   (let ((offset #x40008010))
			     `(
			       (:name ,(format nil "ep~a-dma" e)
				:addr ,(+ offset (* 4 e))
				:reg-access rw
				:type uint16_t
				:fields (
					 (:fname reserved0 :bit 0 :access ro)
					 (:fname dma :bit (13 1) :access rw)
					 (:fname reserved1514 :bit (15 14) :access ro)
					 ))
			       (:name ,(format nil "reserved~8,'0x"
					       (+ offset (+ 2 (* 4 e))))
				:addr ,(+ offset (+ 2 (* 4 e)))
				:reg-access ro
				:type uint16_t))))

		   ,@(loop for e in `(0 1 2 3 4)
			   appending
			   (let ((offset #x40008020))
			     `((:name ,(format nil "ep~a-t-len" e)
				:addr ,(+ offset (* e 4))
				:reg-access rw
				:fields (
					 (:fname t-len :bit (6 0) :access rw :help "transmit length")
					 (:fname reserved0  :bit 7 :access ro )
					 ))
			       (:name ,(format nil "reserved~8,'0x"
					       (+ offset (+ 1 (* e 4)))) 
				:addr ,(+ offset (+ 1 (* e 4)))
				:reg-access ro)
			       (:name ,(format nil "ep~a-ctrl" e) :addr ,(+ offset (+ 2 (* e 4))) 
				:reg-access rw
				:fields (
					 (:fname r-tog :bit 0 :access rw :help "prepared data toggle flag of USB endpoint X receiving (OUT), 0=DATA0, 1=DATA1 ")
					 (:fname t-tog :bit 1 :access rw :help "prepared data toggle flag of USB endpoint X transmittal (IN), 0=DATA0, 1=DATA1 ")
					 (:fname reserved2 :bit 2 :access ro )
					 (:fname auto-tog :bit 3 :access rw :help "automatic toggle after successful transfer completion of on of endpoints 1, 2 or 3")
					 (:fname r-res :bit (5 4) :access rw :help "bitmask for of handshake response type for usb endpoint X, receiving (out)" )
					 (:fname t-res :bit (7 6) :access rw :help "bitmask for of handshake response type for usb endpoint X, transmittal (in)" )
					 
					 ))
			       (:name ,(format nil "reserved~8,'0x" (+ offset (+ 3 (* e 4))))
				:addr ,(+ offset (+ 3 (* e 4)))
				:reg-access ro))))
		   
		   ))
	 (members `(;(max-cores :type int :param t)
		    ;(max-points :type int :param t)
		    #+nil(diagrams :type "std::vector<DiagramData>")
					;(x :type "std::vector<float>")
					;(y :type "std::vector<float>")
		    ;(name-y :type "std::string" :param t)
		    ;(time-points :type "std::deque<float>")
		    )))
    (defun verify-register-addresses (l-regs ds-regs)
      "Verifies that all register addresses in l-regs (used for code generation) match the corresponding addresses in ds-regs (from the datasheet)."
      (loop for l-reg in l-regs
	    do
	       (destructuring-bind (&key name addr reg-access type fields) l-reg
		 
		 (unless (str:starts-with-p "reserved" (format nil "~a" name))
		   
		   (format t "check register ~a~%" name)
		   (let ((ds-reg (find name ds-regs
				       :key #'(lambda (item) (getf item :name))
				       :test #'(lambda (x y) (string= (format nil "~a" x)
								      (format nil "~a" y))))))
		     (unless ds-regs
		       (break "register ~a not found" name))
		     
		     (unless (eq (getf ds-reg :addr)
				 addr)
		       (break "address ~8,'0x of register ~a does not match datasheet value ~8,'0x"
			      addr name (getf ds-reg :addr))))))
	    ))
    (verify-register-addresses l-regs ds-regs)
    (write-class
     :do-format t
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector deque string cstdint)
			(space struct DiagramData (progn
						    "std::string name;"
						    "std::deque<float> values;"
						    )
			       )
			(doc "@brief The DiagramBase class represents a base class for diagrams.
			    
# Description of Interrupt status register

- MASK_UIS_TOKEN identifies the token PID in USB device mode:  
  - 00: OUT packet  
  - 01: SOF packet  
  - 10: IN packet  
  - 11: Idle  
- When MASK_UIS_TOKEN is not idle and RB_UIS_SETUP_ACT is 1:  
  - Process MASK_UIS_TOKEN first  
  - Clear RB_UIF_TRANSFER to make it idle  
  - Then process RB_UIS_SETUP_ACT  
  - Finally, clear RB_UIF_TRANSFER again  
- MASK_UIS_H_RES is valid only in host mode:  
  - For OUT/SETUP token packets from host: PID can be ACK/NAK/STALL or indicate no response/timeout  
  - For IN token packets from host: PID can be data packet PID (DATA0/DATA1) or handshake packet PID

# Description of USB Device Registers

- USB device mode supports 8 bidirectional endpoints: endpoint0 through endpoint7.  
- Maximum data packet length for each endpoint is 64 bytes.  
- Endpoint0: Default endpoint, supports control transmission with a shared 64-byte data buffer for transmission and reception.  
- Endpoint1, endpoint2, endpoint3: Each has a transmission endpoint IN and a reception endpoint OUT with separate 64-byte or double 64-byte data buffers, supporting bulk, interrupt, and real-time/synchronous transmission.  
- Endpoint4, endpoint5, endpoint6, endpoint7: Each has a transmission endpoint IN and a reception endpoint OUT with separate 64-byte data buffers, supporting bulk, interrupt, and real-time/synchronous transmission.  
- Each endpoint has a control register (R8_UEPn_CTRL) and a transmit length register (R8_UEPn_T_LEN) for setting synchronization trigger bit, response to OUT/IN transactions, and length of data to be sent.  
- USB bus pull-up resistor can be software-controlled via USB control register (R8_USB_CTRL) for enabling USB device function; not usable in sleep or power-down mode.  
- In sleep mode, pull-up resistor of DP pin can be enabled via R16_PIN_ANALOG_IE register without being affected.  
- USB protocol processor sets interrupt flag for USB bus reset, suspend, wake-up events, data sending, or receiving; generates interrupt request if enabled.  
- Application can query interrupt flag register (R8_USB_INT_FG) and USB interrupt state register (R8_USB_INT_ST) for processing events based on endpoint number (MASK_UIS_ENDP) and transaction token PID (MASK_UIS_TOKEN).  
- Synchronization trigger bit (RB_UEP_R_TOG) for OUT transactions ensures data packet received matches the endpoint; data is discarded if not synchronous.  
- RB_UEP_AUTO_TOG option available for automatically flipping synchronization trigger bit after successful transmission or reception.  
- Data to be sent/received is stored in their own buffer; sent data length set in R8_UEPn_T_LEN, received data length in R8_USB_RX_LEN, distinguishable by current endpoint number during interrupt.
")
			)
     :implementation-preamble
     `(do0
       
       (include<>
	stdexcept
	format
	)
       )
     :code `(do0
	     
	     (defclass ,name ()
	       "private:"
	       ,@(loop for e in l-regs
		       collect
		       (destructuring-bind (&key name addr (reg-access 'rw) ;(size 1)
					      (type 'uint8_t) fields ) e
			 (declare (ignorable addr reg-access))
			 (if fields
			     `(space
			       union
			       (progn
				 (space ,type reg ,(format nil "; // ~8,'0x" addr))
				 (space struct
					;,(cl-change-case:snake-case (format nil "~a-t" name))
					(progn
					  ,(let ((count-bits 0))
					     `(do0 ,@(loop for field in (reverse fields)
							   collect
							   (destructuring-bind (&key fname bit access help) field
							     (let ((bit-len (if (listp bit)
										(+ 1
										   (- (first bit)
										      (second bit)))
										1)))
							       (incf count-bits bit-len)
							       `(space ,type ,(format nil "~a:~a; // ~a ~@[ ~a~]"
										      (cl-change-case:snake-case (format nil "~a" fname))
										      bit-len
										      access
										      help)))))
						   
						   
						   ,(prog1
							" " ;`(comments ,(format nil "sum of bits = ~a" count-bits))
						      (when fields
							(assert (eq count-bits (ecase type
										 (uint8_t 8)
										 (uint16_t 16))))))
						   
						   )
					     ))
					bit))
			       ,(cl-change-case:snake-case (format nil "~a" name))
			       )
			     `(space ,type ,(cl-change-case:snake-case (format nil "~a" name)))
			     
			     )))
	       "public:" 
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (declare (ignorable type initform))
					    (let (#+nil(nname (intern
							       (string-upcase
								(cl-change-case:snake-case (format nil "~a" name)))))
						  (nname_ (intern
							   (string-upcase
							    (format nil "~a_"
								    (cl-change-case:snake-case (format nil "~a" name)))))))
					      (when param
						nname_))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (declare (ignorable initform))
				       (let (#+nil (nname (intern
							   (string-upcase
							    (cl-change-case:snake-case (format nil "~a" name)))))
					     (nname_ (intern (string-upcase
							      (format nil "~a_"
								      (cl-change-case:snake-case (format nil "~a" name)))))))
					 (when param
					   
					   `(type ,(cond
						     ((and (stringp type)
							   (or (str:starts-with-p "std::vector<" type)
							       (str:starts-with-p "std::deque<" type)
							       (str:starts-with-p "std::array<" type)
							       (str:starts-with-p "std::string" type)))
						      (format nil "const ~a&" type))
						     (t type))
						  ,nname_))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
					(declare (ignorable type initform))
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname ,nname_)) 
					    #+nil (initform
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
				    (declare (ignorable initform param))
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (get (cl-change-case:pascal-case (format nil "get-~a" name)))
					  #+nil (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(defmethod ,get ()
					 (declare (values ,(cond
							     ((and (stringp type)
								   (or (str:starts-with-p "std::vector<" type)
								       (str:starts-with-p "std::deque<" type)
								       (str:starts-with-p "std::array<" type)
								       (str:starts-with-p "std::string" type)))
							      (format nil "const ~a&" type))
							     (t type)))
						  (const))
					 (return ,nname))))))
	       
	       "protected:"
	       
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let (#+nil(nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (cond
					(param `(space ,type ,nname_))
					((and (stringp type)
					      (or (str:starts-with-p "std::vector<" type)
						  (str:starts-with-p "std::deque<" type)
						  (str:starts-with-p "std::array<" type)
						  (str:starts-with-p "std::string" type)))
					 `(space ,type ,nname_ (curly)))
					(t `(space ,type ,nname_ (curly ,initform)))))))))))

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



