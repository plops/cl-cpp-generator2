(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "alexandria")
  (ql:quickload "cl-change-case")
  (ql:quickload "str"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    :format ;; use format (otherwise sstream)
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ;:format
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
			     (:fname sys-ctrl :bit (5 4) :access rw :help "host-mode==0: 00..disable usb device function and disable internal pull-up (can be overridden by dev-pullup-en), 01..enable device fucntion, disable internal pull-up, external pull-up-needed, 1x..enable usb device fucntion and internal 1.5k pull-up, pull-up has priority over pull-down resistor")
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
					 (:fname t-res :bit (7 6) :access rw :help "bitmask for of handshake response type for usb endpoint X, transmittal (in) (see datasheet p. 134)" )
					 
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
			#+nil (do0
      "#ifdef BUILD_FOR_TARGET"
      "#define FMT_THROW panic"
      "#endif")
			
			(include<>	;vector deque
			 string
					;ostream
			 
			 cstdint
			 ;#-format sstream #-format ios
			 ;#+format format
			 ;fmt/format.h
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


I decided to use an anonymous struct inside of the union, so that I
can write usb.int_flag.transfer instead of usb.int_flag.bit.transfer.
For maximum portability and to adhere strictly to the C++ standard,
it's better to name these structs and access them through named
members. However, compilers like GCC and Clang do support anonymous
structs and unions in C++ as an extension.

As this is only an experiment and only has to work for this particular
chip and the GCC compiler, I'm willing to loose compatibility for
convenience.



When using bit fields and packing them into uint8_t, you're unlikely
to encounter alignment issues for this particular use case. However,
when dealing with larger structs or different types, be mindful of
alignment and how different compilers may pack bit fields differently.
I think I will have to look at the compiler output of -Wpadding and do
some testing to verify the memory alignment of this class matches the
registers.


")
			)
     :implementation-preamble
     `(do0
       (space "extern \"C\""
	      (progn
		(include<> CH592SFR.h)))

       (comments "")

       (include "Uart.h")
       
       (include<> 
	array)

       #+nil(include<>
					;sstream
					;ios
	     )

       ,@(loop for (e f) in `((EP0_Databuf ,(+ 64 64 64))
			      (EP1_Databuf ,(+ 64 64))
			      (EP2_Databuf ,(+ 64 64))
			      (EP3_Databuf ,(+ 64 64)))
	     collect
	     `(space extern (__attribute (paren (aligned 4)))
		     ,(format nil "std::array<uint8_t, ~a>" f)
		     ,e))
       )
     :code `(do0
	     
	     (defclass ,name ()
	       "public:"
	       ,@(loop for e in l-regs
		       collect
		       (destructuring-bind (&key name addr (reg-access 'rw) ;(size 1)
					      (type 'uint8_t) fields ) e
			 (declare (ignorable addr reg-access))
			 (let ((struct-name (cl-change-case:pascal-case (format nil "~a" name)))
			       (member-name (cl-change-case:snake-case (format nil "~a" name))))
			  (if fields
			      `(space
				struct ,struct-name
				(progn
				  (space
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
						       )))))) 
				  (defun+ operator= (value)
				    (declare (type ,type value)
					     (values ,(format nil "~a&" struct-name)))
				    (setf reg value)
				    (return *this))
				 #+more
				 (defun+ toString ()
				   (declare (const)
					    (values "std::string"))
				   #+format
				   (return
				     ,(let ((vars-fmt
					      (remove-if #'null
							 (loop for field in (reverse fields)
							       appending
							       (destructuring-bind (&key fname bit access help) field
								 (unless (str:starts-with-p "reserved" (format nil "~a" fname))
								   `(,(format nil "~a~@[ (~a)~]: {} = 0x{:X}" fname (when (eq 'ro access)
														      access))
					; (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								     ))))))
					    (vars-values
					      (remove-if #'null
							 (loop for field in (reverse fields)
							       appending
							       (destructuring-bind (&key fname bit access help) field
								 (unless (str:starts-with-p "reserved" (format nil "~a" fname))
								   `(
								     (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								     (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								     )))))))
					`(std--format
					  (string ,(format nil "~{~a~^,\\n~}"
							   vars-fmt))
					  ,@vars-values)))

				   (do0
				    (let ((out (std--string))))
				    ,(let ((vars-fmt
					     (remove-if #'null
							(loop for field in (reverse fields)
							      appending
							      (destructuring-bind (&key fname bit access help) field
								(unless (str:starts-with-p "reserved" (format nil "~a" fname))
								  `(,(format nil "~a~@[ (~a)~]: {} = 0x{:X}" fname (when (eq 'ro access)
														     access))
					; (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								    ))))))
					   (vars-values
					     (remove-if #'null
							(loop for field in (reverse fields)
							      appending
							      (destructuring-bind (&key fname bit access help) field
								(unless (str:starts-with-p "reserved" (format nil "~a" fname))
								  `(
								    (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								    (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								    )))))))
				       `(fmt--format_to
					 (std--back_inserter out)
					 (string ,(format nil "~{~a~^,\\n~}"
							  vars-fmt))
					 ,@vars-values))
				    (return out))

				   #+nil ;#-format
				   (let ((ss (std--ostringstream)))
				     (<< ss
					 ,@(remove-if #'null
						      (loop for field in (reverse fields)
							    appending
							    (destructuring-bind (&key fname bit access help) field
							      (unless (str:starts-with-p "reserved" (format nil "~a" fname))
								`((string ,(format nil "~a~@[ (~a)~]: " fname (when (eq 'ro access)
														access)))
								  std--dec
								  (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))
								  (string " = 0x")
								  std--hex
								  (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" fname)))))))))
				     (return (ss.str)))))
				,member-name (curly 0))
			      `(space ,type ,(cl-change-case:snake-case (format nil "~a" name))
				      (curly 0))
			      ))))
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


	       
	       
	       (defmethod device_init (ep0_data)
		 (declare (type "uint16_t" ep0_data))
		 (let ((&u (Uart--getInstance))))
		 (comments "the following message takes 47us at 6Mbps (actually 7.4Mbps)")
		 #+nil (u.print (string "Usb device_init ep0_data=0x{:X}")
			  ep0_data)
		 (u.print (string "Usb device_init"))
		 (comments "Reset control register, clear all settings")
		 (setf ctrl.reg 0)
		 (comments "Enable Endpoints 4 (OUT+IN) and 1 (OUT+IN)")
		 (setf ep4_1_mod.ep4_rx_en 1
		       ep4_1_mod.ep4_tx_en 1
		       ep4_1_mod.ep1_rx_en 1
		       ep4_1_mod.ep1_tx_en 1)
		 (comments "Enable Endpoints 2 (OUT+IN) and 3 (OUT+IN)")
		 (setf ep2_3_mod.ep2_rx_en 1
		       ep2_3_mod.ep2_tx_en 1
		       ep2_3_mod.ep3_rx_en 1
		       ep2_3_mod.ep3_tx_en 1)
		 (comments "Set DMA addresses for Endpoints 0, 1, 2 and 3")
		 (setf ep0_dma.dma (static_cast<uint16_t>
				    (reinterpret_cast<uint32_t> (EP0_Databuf.data))) ;ep0_data
		       ep1_dma.dma (static_cast<uint16_t>
				    (reinterpret_cast<uint32_t> (EP1_Databuf.data))) 
		       ep2_dma.dma (static_cast<uint16_t>
				    (reinterpret_cast<uint32_t> (EP2_Databuf.data))) 
		       ep3_dma.dma (static_cast<uint16_t>
				    (reinterpret_cast<uint32_t> (EP3_Databuf.data))))

		 (do0
		  (comments "Configure endpoints, enable automatic ACK on receiving data, and initial NAK on transmitting data")
		  (setf 
		   ep0_ctrl.t_res "0b10" ;; respond to NAK or busy
		   )
		  (setf ep1_ctrl.auto_tog 1
			ep1_ctrl.t_res "0b10")
		  (setf ep2_ctrl.auto_tog 1
			ep2_ctrl.t_res "0b10")
		  (setf ep3_ctrl.auto_tog 1
			ep3_ctrl.t_res "0b10")
		  (setf 
			ep4_ctrl.t_res "0b10"))

		 (comments "clear device address")
		 (setf dev_ad.reg 0)

		 (comments "Enable usb device pull-up resistor, DMA, and interrupts")
		 (setf ctrl.dma_en 1
		       ctrl.int_busy 1
		       ctrl.sys_ctrl "0b10"
		       ctrl.low_speed 0
		       ctrl.host_mode 0)
		 (comments "Disable analog features on USB pins")
		 (setf R16_PIN_ANALOG_IE
		       (or R16_PIN_ANALOG_IE
			   RB_PIN_USB_IE
			   RB_PIN_USB_DP_PU))
		 (comments "Clear interrupt flags")
		 (setf int_flag.reg (hex #xff))
		 (comments "Power on the USB port")
		 (setf port_ctrl.port_en 1
		       port_ctrl.pd_dis 1
		      ; port_ctrl.low_speed 0
		      ; port_ctrl.hub0_reset 0
		       )
		 (comments "Enable interrupts for suspend, bus reset, and data transfers")
		 (setf int_en.suspend 1
		       int_en.transfer 1
		       int_en.bus_reset 1)

		 
		 )
	       
	       "private:"
	       
	       
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

  
  (let* ((name `UsbDeviceDescriptor)
	 (members `((bLength :type "const uint8_t" :initform 18)
		    (bDescriptorType :type "const  uint8_t" :initform 1)
		    (bcdUSB :type "const uint16_t" :initform (hex #x0200))
		    (bDeviceClass :type uint8_t :param t)
		    (bDeviceSubClass :type uint8_t :param t)
		    (bDeviceProtocol :type uint8_t :param t)
		    (bMaxPacketSize :type "const  uint8_t" :initform 64)
		    (idVendor :type uint16_t :param t)
		    (idProduct :type uint16_t :param t)
		    (bcdDevice :type uint16_t :param t)
		    (iManufacturer :type "const  uint8_t" :initform 0)
		    (iProduct :type "const uint8_t" :initform 0)
		    (iSerialNumber :type "const  uint8_t" :initform 0)
		    (bNumConfigurations :type uint8_t :param t)
		    
		    )))
    (write-class
     :do-format t
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name 
     :headers `()
     :header-preamble `(do0
			(include<> ;vector deque
			 string
				   cstdint)
			
			(doc "
**Device Descriptor**

* **Represents the entire USB device:**  One device = one descriptor.
* **Key Device Information:**
    * **USB Version Supported:**  Device's USB spec compliance (e.g., 2.0, 1.1)
    * **Maximum Packet Size (Endpoint 0):**  Largest data unit for default endpoint.
    * **Vendor ID:**  USB-IF assigned ID for the device's manufacturer.
    * **Product ID:** Manufacturer-assigned ID for the specific device. 
    * **Number of Configurations:** How many ways the device can be configured.

**Understanding Fields**

* **bcdUSB:** Binary-coded decimal for USB version (e.g., 0x0200 = USB 2.0)
* **bDeviceClass, bDeviceSubClass, bDeviceProtocol:**  Codes used to find the appropriate device driver. Often more specific codes are defined at the interface level.
* **bcdDevice:** Device version number set by the developer.
* **iManufacturer, iProduct, iSerialNumber:**  Indexes pointing to optional string descriptors for additional human-readable information.
* **bNumConfigurations:**  Indicates the total number of potential device setups. 
")
			)
     :implementation-preamble
     `(do0
       (comments "")

       (do0
      "#ifdef BUILD_FOR_TARGET"
      "#define FMT_THROW panic"
      "#endif")
       
       (include<>
					;stdexcept
					;format
					;cstdint
	#-format sstream
	#+format format
	)
       
       )
     :code `(do0
	     
	     (defclass ,name ()
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
		 (static_assert (== 18
				    (sizeof UsbDeviceDescriptor)
				    ))
		 )
	       #+more (defmethod toString ()
		 (declare (const)
			  (values "std::string"))
		 #+format (return
		  (std--format
		   (string ,(format nil "~{~a~^,\\n~}"
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (format nil "~a: {} = 0x{:X}" name)))))
		   ,@(loop for e in members
			       appending
			       (destructuring-bind (name &key type param (initform 0)) e
				 `((static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   )))))
		 
		 #-format
			(let ((ss (std--ostringstream)))
		   (<< ss
		       ,@(loop for e in members
			       appending
			       (destructuring-bind (name &key type param (initform 0)) e
				 `((string ,(format nil "~a: " name))
				   std--dec
				   (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   (string " = 0x")
				   std--hex
				   (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   (string "\\n")))))
		   (return (ss.str))))
	       (doc "
@brief isValid() checks if the const members b_length,
b_descriptor_type, bcd_usb, and b_max_packet_size have the expected
values. These values are defined based on the USB specification that
the UsbDeviceDescriptor is designed to represent. In a real-world
scenario, these checks ensure that the hardcoded values haven't been
tampered with or incorrectly modified due to a programming error or
memory corruption.

This method shall be used if you cast an arbitrary uint8_t array to
UsbDeviceDescriptor. 
")
	       (defmethod isValid ()
		 (declare (const)
			  (values bool))
		 (when (logior
			,@(remove-if #'null
				     (loop for e in members
					   collect
					   (destructuring-bind (name &key type param (initform 0)) e
					     (let (#+nil(nname (cl-change-case:snake-case (format nil "~a" name)))
						   (nname (format nil "~a" (cl-change-case:snake-case (format nil "~a" name)))))
					       (cond
						 ((and (stringp type)
						       (str:starts-with-p "const" type)
						       )
						  `(!= ,initform ,nname))))))))
		   (return false))
		 (return true)
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
	       
	       "private:"
	       
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let (#+nil(nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname (format nil "~a" (cl-change-case:snake-case (format nil "~a" name)))))
				      (cond
					(param `(space ,type ,nname))
					((and (stringp type)
					      (or (str:starts-with-p "std::vector<" type)
						  (str:starts-with-p "std::deque<" type)
						  (str:starts-with-p "std::array<" type)
						  (str:starts-with-p "std::string" type)))
					 `(space ,type ,nname (curly)))
					(t `(space ,type ,nname (curly ,initform)))))))))))

    )
  
  (let* ((name `UsbConfigurationDescriptor)
	 (members `((bLength :type uint8_t :param t)
		    (bDescriptorType :type "const  uint8_t" :initform 2)
		    (wTotalLength :type uint16_t :param t)
		    (bNumInterfaces :type uint8_t :param t)
		    (bConfigurationValue :type uint8_t :param t)
		    (iConfiguration :type "const uint8_t" :initform 0)
		    (bmAttributes :type uint8_t :param t)
		    (bMaxPower :type uint8_t :param t)
		    
		    )))
    (write-class
     :do-format t
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> ;vector deque string
				   cstdint
				   ostream)
			
			(doc "
**Configuration Descriptor Summary**

* **Device configurations:** A USB device can have multiple configurations, although most devices only have one.
* **Configuration details:** The configuration descriptor specifies power consumption, interfaces, and transfer mode.
* **Configuration selection:** The host selects a configuration using a `SetConfiguration` command.

**Descriptor Fields Explained**


| Field               | Description                                                                  |
|---------------------|------------------------------------------------------------------------------|
| bLength             | Size of the descriptor in bytes.                                             |
| bDescriptorType     | Constant value indicating a configuration descriptor (0x02).                 |
| wTotalLength        | Total length in bytes of data returned, including all following descriptors. |
| bNumInterfaces      | Number of interfaces included in the configuration.                          |
| bConfigurationValue | Value used to select this configuration.                                     |
| iConfiguration      | Index of a string descriptor describing the configuration.                   |
| bmAttributes        | Bitmap containing power configuration details see below                      |
| bMaxPower           | Maximum power consumption from the bus in 2mA units (maximum of 500mA).      |

bmAttributes:
    * D7: Reserved (set to 1 for USB 1.0 bus-powered devices).
    * D6: Self-powered.
    * D5: Remote wakeup capable.
    * D4..0: Reserved (set to 0). 

I think string descriptors are optional, so for now I will always keep string indices 0.

")
			)
     :implementation-preamble
     `(do0

       (do0
      "#ifdef BUILD_FOR_TARGET"
      "#define FMT_THROW panic"
      "#endif")
       
       (include<>
					;stdexcept
	#+format format
					;cstdint
	#-format sstream
	)
       
       )
     :code `(do0
	     
	     (defclass ,name ()
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
	       #+more (defmethod toString ()
		 (declare (const)
			  (values "std::string"))
			#+format (return
		  (std--format
		   (string ,(format nil "~{~a~^,\\n~}"
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (format nil "~a: {} = 0x{:X}" name)))))
		   ,@(loop for e in members
			       appending
			       (destructuring-bind (name &key type param (initform 0)) e
				 `((static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   )))))
			#-format
			(let ((ss (std--ostringstream)))
		   (<< ss
		       ,@(loop for e in members
			       appending
			       (destructuring-bind (name &key type param (initform 0)) e
				 `((string ,(format nil "~a: " name))
				   std--dec
				   (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   (string " = 0x")
				   std--hex
				   (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
				   (string "\\n")))))
		   (return (ss.str))))
	       (defmethod isValid ()
		 (declare (const)
			  (values bool))
		 (when (logior
			,@(remove-if #'null
				     (loop for e in members
					   collect
					   (destructuring-bind (name &key type param (initform 0)) e
					     (let (#+nil(nname (cl-change-case:snake-case (format nil "~a" name)))
						   (nname (format nil "~a" (cl-change-case:snake-case (format nil "~a" name)))))
					       (cond
						 ((and (stringp type)
						       (str:starts-with-p "const" type)
						       )
						  `(!= ,initform ,nname))))))))
		   (return false))
		 (return true)
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
	       
	       "private:"
	       
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let (#+nil(nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname (format nil "~a" (cl-change-case:snake-case (format nil "~a" name)))))
				      (cond
					(param `(space ,type ,nname))
					((and (stringp type)
					      (or (str:starts-with-p "std::vector<" type)
						  (str:starts-with-p "std::deque<" type)
						  (str:starts-with-p "std::array<" type)
						  (str:starts-with-p "std::string" type)))
					 `(space ,type ,nname (curly)))
					(t `(space ,type ,nname (curly ,initform)))))))))))

    )

  (let* ((name `Uart)
	 (members `(
		    ;; fixme: mutex doesn't compile for bare-metal risc-v
		    ;(mutex :type "std::mutex" :param nil :internal t)
		    
		    )))
    (write-class
     :do-format t
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector
					;deque
					;string
				   ;cstdint
				   ;ostream
				   ;mutex
				   
				   )
			(do0
			 #+nil
			 (do0
			  "#ifdef BUILD_FOR_TARGET"
			  "#define FMT_THROW panic"
			  "#endif")
			 (include<> fmt/format.h
				    ))
			)
     :implementation-preamble
     `(do0
       (comments "implementation"       )
       #+nil
       (include<>			;cassert
	cstddef)
       (space extern "\"C\""
	      (progn
	       	(include<> CH59x_common.h
			   ;CH59x_uart.h
			   ;CH59x_gpio.h
			   )))

     
              
       )
     :code `(do0
	     (doc "
- The constructor of Uart is made private to prevent direct
  instantiation.

- A static method getInstance() is provided to get the singleton
  instance.

- Copy constructor and copy assignment operator are deleted to prevent copying.

- Not working on bare metal risc-v yet: A std::mutex named mutex is
  added to protect critical sections within the print method. This
  mutex is locked using std::lock_guard before accessing shared
  resources.

- Please note, using a mutex in a high-frequency logging or in
  interrupt context can lead to performance bottlenecks or deadlocks
  if not handled carefully. Always consider the specific requirements
  and constraints of your embedded system when introducing
  thread-safety mechanisms.
")
	     (defclass ,name ()
	       "private:" 
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0) internal) e
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
				     (destructuring-bind (name &key type param (initform 0) internal) e
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
				      (destructuring-bind (name &key type param (initform 0) internal) e
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
		 (do0
		  (comments  "This will configure UART1 to send and receive at 115200 baud:")
		  (doc "up to 6Mbps is possible. fifo can store 8 bytes")
		  (GPIOA_SetBits GPIO_Pin_9)
		  (GPIOA_ModeCfg GPIO_Pin_8 GPIO_ModeIN_PU)
		  (GPIOA_ModeCfg GPIO_Pin_9 GPIO_ModeOut_PP_5mA)
		  ;(UART1_DefInit)
		  (comments "logic analyzer can decode 1'000'000 setting with 938'000 Hz")
		  (comments "logic analyzer can decode 6'000'000 setting with 7'400'000 Hz")
		  (UART1_BaudRateCfg "6'000'000"
					;"115'200"
				     )
		  (comments "clear and enable fifos")
		  (setf R8_UART1_FCR (or (<< 2 6)
					     RB_FCR_TX_FIFO_CLR
					     RB_FCR_RX_FIFO_CLR
					     RB_FCR_FIFO_EN))
		  (comments "8-bit word size, odd parity, 2 stop bits")
		  (setf R8_UART1_LCR (or RB_LCR_WORD_SZ
					 RB_LCR_PAR_EN
					 RB_LCR_STOP_BIT))
		  (comments "enable tx interrupt")
		  (setf R8_UART1_IER RB_IER_TXD_EN)
		  (comments "pre-divisor latch byte (7-bit value), don't change clock, leave 1")
		  (setf R8_UART1_DIV 1)
		  )
		 )
	       "public:"
	       (defun+ getInstance ()
		 (declare (values "static Uart&"))
		 (space static Uart instance)
		 (return instance))
	       (comments "Delete copy constructor and assignment operator")
	       (= (Uart "Uart const&") delete)
	       (= "Uart& operator=(Uart const&)" delete)
	       #+more (defmethod toString ()
			(declare (const)
				 (values "std::string"))
			#+format (return 
				   (std--format
				    (string ,(format nil "~{~a~^,\\n~}"
						     (loop for e in members
							   collect
							   (destructuring-bind (name &key type param (initform 0)) e
							     (format nil "~a: {} = 0x{:X}" name)))))
				    ,@(loop for e in members
					    appending
					    (destructuring-bind (name &key type param (initform 0)) e
					      `((static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
						(static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
						)))))
			#-format
			(let ((ss (std--ostringstream)))
			  (<< ss
			      ,@(loop for e in members
				      appending
				      (destructuring-bind (name &key type param (initform 0)) e
					`((string ,(format nil "~a: " name))
					  std--dec
					  (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
					  (string " = 0x")
					  std--hex
					  (static_cast<int> ,(cl-change-case:snake-case (format nil "~a" name)))
					  (string "\\n")))))
			  (return (ss.str))))
	       (space "template <typename... Args>"
		      (defun+ print (fmt args)
			(declare (type "fmt::format_string<Args...>" fmt)
				 (type Args&&... args))
			(let ((ostr (std--vector<uint8_t>)))
			  (comments "Use format_to with a back_inserter to append formatted output to the vector")
			  (fmt--format_to (std--back_inserter ostr)
					  fmt
					  "std::forward<Args>(args)...")
			  (SendString (ostr.data)
				      (static_cast<uint16_t> (ostr.size))))
			))
	       (doc "Overload for const char pointer")
	       (defmethod print (str)
		 (declare (type "const char*" str))
		 (let ((n (strlen str)))
		   #+nil
		   (assert (logand (<= n std--numeric_limits<uint16_t>--max)
				   (string "String length exceedds uint16_t range"))))
		 (SendString (reinterpret_cast<uint8_t*> (const_cast<char*> str))
			     (static_cast<uint16_t> n)))
	       (doc "Overload for string literals (will not call strlen for known strings)")
	       (space "template <std::size_t N>"
		      (defun+ print ([N] )
			(declare (type "const char (&str)" [N])
				 )
			(comments "N includes the null terminator, so we subtract 1 1t oget the actual string length")
			(SendString (reinterpret_cast<uint8_t*> (const_cast<char*> str))
				    (static_cast<uint16_t> (- N 1)))))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform internal) e
				    (declare (ignorable initform param))
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (get (cl-change-case:pascal-case (format nil "get-~a" name)))
					  #+nil (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (unless internal
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
					   (return ,nname)))))))
	       
	       "private:"

	       (defmethod SendString (buf len)
		 (declare (type uint8_t* buf)
			  (type uint16_t len))
		 (comments "FIXME: hold mutex here")
		 (UART1_SendString buf len)
		 )
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0) internal) e
				    (let (#+nil(nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname (format nil "~a" (cl-change-case:snake-case (format nil "~a" name)))))
				      (cond
					(param `(space ,type ,nname))
					((and (stringp type)
					      (or (str:starts-with-p "std::vector<" type)
						  (str:starts-with-p "std::deque<" type)
						  (str:starts-with-p "std::array<" type)
						  (str:starts-with-p "std::string" type)))
					 `(space ,type ,nname (curly)))
					(t `(space ,type ,nname (curly ,initform)))))))))))

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
      cassert
      algorithm
      )

     
     (include Ch592UsbRegisters.h
	      UsbDeviceDescriptor.h
	      UsbConfigurationDescriptor.h
	      Uart.h)


     (do0
      "#ifdef BUILD_FOR_TARGET"
      (space extern "\"C\""
	     (progn
	       
	       (include<> 
		CH59x_common.h)))
      "#else"
      (include<> format
		 iostream)
      "#endif")
     
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
			    (EP1_Databuf ,(+ 64 64))
			    (EP2_Databuf ,(+ 64 64))
			    (EP3_Databuf ,(+ 64 64))) 
	     collect
	     `(space (__attribute (paren (aligned 4)))
		     ,(format nil "std::array<uint8_t, ~a>" f)
		     ,e))

     (do0
      "#ifdef BUILD_FOR_TARGET"
      (do0
       (space constexpr uintptr_t (= c_USB_BASE_ADDR (hex #x40008000)))
       (space Ch592UsbRegisters& (= usb (deref (new (paren (reinterpret_cast<void*> c_USB_BASE_ADDR)))
					       ))
	      Ch592UsbRegisters)) 
      (do0

       (comments "overview usb https://www.beyondlogic.org/usbnutshell/usb3.shtml")
       (defun USB_DevTransProcess2 ()
	(let ((&u (Uart--getInstance))))

	 (cond
	   ,@(loop for (e f) in `((transfer T)
				  (bus_reset R)
				  (suspend S))
		   collect
		   `((dot usb int_flag ,e)
		     (setf (dot usb int_flag ,e) 1
			   )
		     (u.print (string ,f)))))
	 #+nil
	 (when usb.int_flag.transfer
	   (when (!= (hex #b11) usb.int_status.token )
	     (cond
	       ((== (hex #b01 ) usb.int_status.token)
		(comments "usb token in")
		(case usb.int_status.endp
		  (0
		   (u.print (string "usb token in EP0"))
		   (case SetupReqCode
		     (USB_GET_DESCRIPTOR
		      (u.print (string "get descriptor")))
		     (USB_SET_ADDRESS
		      (u.print (string "set address")))
		     (t
		      (comments default))))
		  (1 (u.print (string "usb token in EP1")))
		  (2 (u.print (string "usb token in EP2")))
		  (3 (u.print (string "usb token in EP3")))
		  (4 (u.print (string "usb token in EP4")))))
	       ((== (hex #b00) usb.int_status.token)
		(u.print (string "usb token out"))
		(case usb.int_status.endp
		  (0 
		   (u.print (string "token out EP0")))
		  (1
		   (u.print (string "token out EP1")))
		  (2
		   (u.print (string "token out EP2")))
		  (3
		   (u.print (string "token out EP3")))
		  (4
		   (u.print (string "token out EP4"))))
		)))
	   (u.print (string "clear interrupt by writing to flag"))
	   (setf usb.int_flag.transfer 1)))

       #+nil
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
	      )))))
      "#else"
      (do0
       (space Ch592UsbRegisters& (= usb (deref (new)))
	      Ch592UsbRegisters))
      "#endif")
     

     (space
      "extern \"C\""
      (progn

	
	(do0
	 (doc "

__INTERRUPT is defined with __attribute__((interrupt('WCH-Interrupt-fast'))). This likely indicates a specialized, 'fast' interrupt mechanism specific to your compiler or microcontroller (WCH).


The compiler attribute __attribute__((section('.highcode'))) will be assigned to the __HIGH_CODE macro. This attribute likely instructs the compiler to place functions or code blocks marked with __HIGH_CODE into a special memory section named '.highcode' (possibly a faster memory region).

Here is a post about fast interrupts on WCH https://www.reddit.com/r/RISCV/comments/126262j/notes_on_wch_fast_interrupts/
")
	 (space	 ;(__attribute__ (paren (interrupt (string "user" ))))
	  (__attribute__ (paren interrupt))
	  ;__INTERRUPT
	  __HIGH_CODE
	  (defun USB_IRQHandler ()
	    (comments "Handle interrupts coming from the USB Peripheral")
	    #+nil (let ((&u (Uart--getInstance)))
	      (u.print (string "usb_irq")))
	    (USB_DevTransProcess2))))

	
	(space 
	 (__attribute__ (paren interrupt))
					;__INTERRUPT
	 __HIGH_CODE
	 (defun TMR0_IRQHandler ()
	   (comments "Check if the TMR0_3_IT_CYC_END interrupt flag is set")
	   (when (TMR0_GetITFlag TMR0_3_IT_CYC_END)
	     (comments "Clear interrupt flag")
	     (TMR0_ClearITFlag TMR0_3_IT_CYC_END)

	     ;#+nil
	     (do0
	      (comments "Print a T character on the Uart (if FIFO isn't full)")
	      (unless (== R8_UART1_TFC
			  UART_FIFO_SIZE)
		(setf R8_UART1_THR (char "t"))))
	     
	     #+nil
	     (let ((&u (Uart--getInstance)))
	       (u.print (string "timer"))))
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


     (doc
      "
Here's a bullet list summary of the essential concepts regarding USB Protocols:

**Understanding USB Protocols**

* **Layered Structure:** USB protocols operate in layers, simplifying design. Higher layers are more relevant for users, with lower layers handled by USB controllers.
* **Transactions:** USB data exchange occurs in transactions with the following components:
    * Token Packet (header)
    * Optional Data Packet (payload)
    * Status Packet (acknowledgment/error correction)

**Key Packet Fields:**

* **Sync:** Synchronizes transmitter and receiver clocks.
* **PID (Packet ID):**  Identifies the packet type (token, data, handshake, etc.).
* **ADDR:** Specifies the destination device address.
* **ENDP:** Identifies the specific endpoint on the device.
* **CRC:** Error detection mechanism.
* **EOP:** Marks the end of a packet.

**Packet Types**

* **Token:** Indicates transaction type (IN, OUT, SETUP)
* **Data:** Carries the actual data payload (DATA0, DATA 1, etc.).
* **Handshake:** Acknowledges transactions or signals errors (ACK, NAK, STALL)
* **Start of Frame (SOF):** Sent periodically to mark time intervals.

**USB Functions and Devices**

* **USB Function:** A USB device with a specific capability (printer, scanner, etc.). Note that this can include host controllers or hubs as well. 
* **Endpoints:** Points on a USB function where data is sent or received. Endpoint 0 is mandatory for control/status.
* **Pipes:** Logical connections between the host software and device endpoints, defining transfer parameters.

**Key Points**

* USB is host-centric; the host initiates all transactions.
* Most USB controllers handle low-level protocol implementation.
* Understanding endpoints and pipes is crucial for USB device design. 


")
     
     (doc
      "
**Control Transfers: Purpose and Characteristics**

* **Function:** Used for device setup (enumeration), command & status operations
* **Initiation:** Always started by the host computer
* **Nature:** Bursty, random packets 
* **Error Handling:** Utilize a best-effort delivery approach
* **Packet Sizes:**
    * Low-speed: 8 bytes
    * Full-speed: 64 bytes
    * High-speed: 8, 16, 32, or 64 bytes

**Stages of a Control Transfer**

1. **Setup Stage:**
   * Host sends a setup token (address, endpoint)
   * Host sends a data packet (DATA0) containing the setup details
   * Device acknowledges (ACK) if data is received successfully

2. **Optional Data Stage:**
   * One or more IN/OUT transactions depending on data direction
   * Data is sent in chunks matching the maximum packet size
   * Device can signal readiness (ACK), temporary unavailability (NAK), or an error (STALL)

3. **Status Stage:**
    *  Direction dictates the status reporting procedure:
       * IN Transfer: Host acknowledges data, device reports status
       * OUT Transfer: Device acknowledges data, host checks status

**Big Picture Example: Requesting a Device Descriptor**

1. **Setup:** Host sends a setup token, then a DATA0 packet with the descriptor request. Device acknowledges.
2. **Data:** Host sends IN tokens. Device sends the descriptor in chunks, with the host acknowledging each chunk.
3. **Status:** Host sends a zero-length OUT packet to signal success, the device responds to confirm its own status.
")

     (do0
      "#ifdef BUILD_FOR_TARGET"
      (defun main ()
	(declare (values int))
	
	
	
	(SetSysClock CLK_SOURCE_PLL_60MHz)

	(let ((&u (Uart--getInstance))))
	(u.print (string "main"))

	
	(do0
	 (comments "Enable timer with 100ms period") 
	 (TMR0_TimerInit FREQ_SYS/10)
	 (TMR0_ITCfg ENABLE TMR0_3_IT_CYC_END)
	 #+nil
	 (let ((tmr0_addr (reinterpret_cast<uint32_t*> (hex #x40))))
	   (setf *tmr0_addr (reinterpret_cast<uint32_t> TMR0_IRQHandler))
	   (u.print (string "tmr0=0x{:X}")
		    *tmr0_addr))
	 
	 (PFIC_EnableIRQ TMR0_IRQn))
	

	#+nil  (setf pEP0_RAM_Addr (EP0_Databuf.data))

	
	(do0
	 (let ((&dev (deref ("reinterpret_cast<const UsbDeviceDescriptor*>" (DevDescr.data))))
	       (&cfg (deref ("reinterpret_cast<const UsbConfigurationDescriptor*>" (CfgDescr.data))))))
					; (dev.isValid)
					; (cfg.isValid)
		 
	 (do0

	  (usb.device_init (static_cast<uint16_t>
			    (reinterpret_cast<uint32_t> (EP0_Databuf.data))))
	  (comments "Enable the interrupt associated with the USB peripheral.")
	  (PFIC_EnableIRQ USB_IRQn)
	  (u.print (string "usb_irq=on"))))
	
	#+nil,@(loop for e in `(int_en.suspend
			   int_en.transfer
			   int_en.bus_reset)
			 collect
			 `(u.print (string ,(format nil "~a=0x{:X}\\r\\n" e))
				   (static_cast<int> (dot usb ,e))))
	(u.print (string "start USB_DeviceInit\\r\\n"))
	(USB_DeviceInit)
	
	(while 1
	       (comments "inifinite loop")
	       (__nop)
	       #+nil
	       ,(let ((main-delay 7))
		  `(do0
		    (u.print (string ,(format nil "AAAA~a" main-delay)))
		    (mDelaymS ,main-delay)
		    (u.print (string ,(format nil "MAIN~a" main-delay)))))
					;(u.print (string "hello"))
	       ))
      "#else"
      (defun main ()
	(declare (values int))
	

       

	#+nil(usb.device_init (static_cast<uint16_t>
			       (reinterpret_cast<uint32_t> (EP0_Databuf.data))))

	(let ((&dev (deref ("reinterpret_cast<const UsbDeviceDescriptor*>" (DevDescr.data))))
	      (&cfg (deref ("reinterpret_cast<const UsbConfigurationDescriptor*>" (CfgDescr.data))))))
	(dev.isValid)
	#+more ,(lprint :vars `((dev.toString)))
	(cfg.isValid)
	#+more ,(lprint :vars `((cfg.toString))))
      "#endif")
     


     

     #+nil(defun DevEP1_OUT_Deal (l)
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



