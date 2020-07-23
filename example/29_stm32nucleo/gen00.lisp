(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `(:dac1
				     :adc1)))

(setf *features* (set-difference *features*
				 '(;:dac1
				   ;:adc1
				   )))

(progn
  (defparameter *source-dir* #P"example/29_stm32nucleo/source/")
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  
 
  (progn
    
    ;; the cube ide encloses areas with comments like 
    ;; /* USER CODE BEGIN SysInit */ .. /* USER CODE END SysInit */
    ;;                    ^^^^^^^ --- part-name
    ;; a part is a piece of user code that will be inserted between two part-name comments
    
    (defparameter *parts* nil) 
    (defun logprint (msg &optional rest)
      `(do0
	" "
	#-nolog
	(do0
					;("std::setprecision" 3)
	 (<< "std::cout"
	     ;;"std::endl"
	     ("std::setw" 10)
	     (dot ("std::chrono::high_resolution_clock::now")
		  (time_since_epoch)
		  (count))
					;,(g `_start_time)
	     
	     (string " ")
	     ("std::this_thread::get_id")
	     (string " ")
	     __FILE__
	     (string ":")
	     __LINE__
	     (string " ")
	     __func__
	     (string " ")
	     (string ,msg)
	     (string " ")
	     ,@(loop for e in rest appending
		    `(("std::setw" 8)
					;("std::width" 8)
		      (string ,(format nil " ~a='" (emit-c :code e)))
		      ,e
		      (string "'")))
	     "std::endl"
	     "std::flush"))))
    
    (defun define-part (args)
      (destructuring-bind (file part-name part-code) args
	(push `(:name ,part-name :file ,file :code ,part-code)
	      *parts*))))
  (let ((n-channels 2)
	(n-tx-chars 128)
	(n-dac-vals 4096))
    
    (progn
      (define-part
	 `(main.c Includes
		  (include <stdio.h>)))
      (define-part
	 `(main.c PV
		  (let (#+adc1 (value_adc)
			#+dac1 (value_dac)
			(BufferToSend))
		    (declare (type (array uint16_t ,n-channels) value_adc)
			     (type (array uint16_t ,n-dac-vals) value_dac)
			     (type (array uint8_t ,n-tx-chars) BufferToSend)))))
      (let ((l `((ADC
		  (ConvHalfCplt
		   Error
		   ConvCplt))
		 (UART (Error TransmitCplt AbortOnError))
		 (DAC (Error ConvCplt ConvHalfCplt) :channels (Ch1 Ch2)))))
	(define-part
	    ;; USE_HAL_UART_REGISTER_CALLBACKS
	    ;; should be  defined to 0 in  stm32l4xx_hal_conf.h but it is not there
	    `(main.c 0
		     (do0
		      ,@(loop for e in l appending
			    (destructuring-bind (module irqs &key (channels `(""))) e
			      (loop for irq in irqs append
				   (loop for ch in channels collect
				    `(defun ,(format nil "HAL_~a_~aCallback~a" module irq ch)
					 (arg)
				       (declare (type ,(format nil "~a_HandleTypeDef*" module)
						      arg))
				       (let ((huart2))
					 (declare (type "extern UART_HandleTypeDef" huart2))
					 ,(let ((report (format nil "~a ~a ~a\\r\\n" module irq ch)))
					    `(HAL_UART_Transmit_DMA &huart2 (string ,report)
								    ,(length report))
					    #+nil `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,report)
											     ,(length report)))
						     (Error_Handler)))))))))))))
      (define-part 
	  `(main.c 2
		   (do0
		    #+dac1 (do0
			    ,(let ((report "call HAL_DAC_MspInit\\r\\n"))
			      `(HAL_UART_Transmit_DMA &huart2 (string ,report)
						     ,(length report)))
			    (HAL_DAC_MspInit &hdac1)
			    (HAL_DAC_Start &hdac1 DAC_CHANNEL_1)
				(HAL_DAC_Start_DMA &hdac1 DAC_CHANNEL_1 (cast "uint32_t*" value_dac) ,n-dac-vals
						   DAC_ALIGN_12B_R))
		    #+adc1 (do0 (HAL_ADCEx_Calibration_Start &hadc1 ADC_SINGLE_ENDED)
				(HAL_ADC_Start_DMA &hadc1 (cast "uint32_t*" value_adc) ,n-channels)
				))))
      (define-part 
	  `(main.c 3
		   (do0
		    #+dac1 (do0
					;(HAL_DAC_SetValue &hdac1 DAC_CHANNEL_1 DAC_ALIGN_12B_R value_dac)
			    (progn
				(let ((count))
				  (declare (type "static int" count))
				  (incf count)
				  (when (<= ,(expt 2 12) count)
				    (setf count 0))
				  (setf (aref value_dac count) count)
				  #+nil (if (< value_dac ,(- (expt 2 12) 1))
				     (incf value_dac)
				     (setf value_dac 0))
				  (HAL_Delay 1)
				  (progn
		      #+nil ,(let ((l `(#+dac1 (dac (aref value_dac count))
					 #+adc1 (adc0  ;USE_HAL_UART_REGISTER_CALLBACKS
						      (aref value_adc 0)
						      )
					 #+adc1 (adc1  ;USE_HAL_UART_REGISTER_CALLBACKS
						      (aref value_adc 1)
						      ))))
			 `(let ((n (snprintf (cast int8_t* BufferToSend)
					    ,n-tx-chars
					    (string ,(format nil "~{~a:%d ~}\\r\\n" (mapcar #'first l)))
					    ,@(mapcar #'second l))))
			   (declare (type int n))
			   (unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 BufferToSend n))
			     (Error_Handler))))))))
		    

		    
		    "}"
		    
		    )))
      (let ((l `(,@(loop for e in `(USART2 DMA1_Channel7
					   DMA1_Channel1
					   DMA1_Channel3
					   TIM6_DAC
					   (SysTick :modulo 1000) ;; only show every 1000th interrupt
					   PendSV DebugMonitor SVCall
					   UsageFault BusFault MemoryManagement HardFault
					   NonMaskableInt)
		      collect
			(if (listp e)
			    (destructuring-bind (name &key (modulo 1)) e
				(list (format nil "~a_IRQn 0" name)
				      modulo))
			    (list (format nil "~a_IRQn 0" e)
				  1)))
		   )))
	(loop for (e modulo) in l do
	 (define-part 
	     `(stm32l4xx_it.c
	       ,e
	       (do0
		,(if (eq modulo 1)
		     `(do0
		       (HAL_UART_Transmit_DMA &huart2 (string ,(format nil "~a\\r\\n" e))
									,(+ 2 (length e)))
		       #+nil `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,(format nil "~a\\r\\n" e))
									,(+ 2 (length e))))
				(Error_Handler)))
		     `(progn
			(let ((count 0))
			  (declare (type "static int" count))
			  (incf count)
			  (when (== 0 (% count ,modulo))
			    ,(let ((report (format nil "~a#~a\\r\\n" e modulo)))
			       `(HAL_UART_Transmit_DMA &huart2 (string ,report)
									,(length report))
			       #+nil `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,report)
									,(length report)))
				(Error_Handler)))))))
		)))))

      (let ((l `(,@(loop for e in `(USART2 DAC1 ADC1)
		      appending
			(list ;; MSP means mcu support package
			 (format nil "~a_MspInit 1" e)
			 (format nil "~a_MspDeInit 1" e)
			 ))
		   )))
	(loop for e in l do
	 (define-part 
	     `(stm32l4xx_hal_msp.c
	       ,e
	       (progn
		 (let ((huart2))
		   (declare (type "extern UART_HandleTypeDef" huart2))
		  ,(let ((report (format nil "~a\\r\\n" e)))
		     `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,report)
								,(length report)))
			(Error_Handler))))
		
		))))))
    
    (loop for e in *parts* and i from 0 do
	 (destructuring-bind (&key name file code) e
	   ;; open the file that we will modify
	   (let* ((full-fn (format nil "/home/martin/STM32CubeIDE/workspace_1.3.0/nucleo_l476rg_dac_adc_loopback/Core/Src/~a" file))
		  (a (with-open-file (s full-fn
				       :direction :input)
		      (let ((a (make-string (file-length s))))
			(read-sequence a s)
			a))))
	     (let* ((start-comment (format nil "/* USER CODE BEGIN ~a */" name))
		    (end-comment (format nil "/* USER CODE END ~a */" name))
		    ;; escape * characters to convert c comment to regex
		    (regex (format nil "~a.*~a"
				   (regex-replace-all "\\*" start-comment "\\*")
				   (regex-replace-all "\\*" end-comment "\\*")))
		    ;; now use the regex to replace the text between the comments
		    (new (cl-ppcre:regex-replace (cl-ppcre:create-scanner regex :single-line-mode t)
						 a
						 (format nil "~a~%~a~%~a" start-comment
							 (emit-c :code code)
							 end-comment
							 ))))
	       (with-open-file (s full-fn ;"/dev/shm/o.c"
				  :direction :output :if-exists :supersede :if-does-not-exist :create)
		 (write-sequence new s))
	       ))))))

;; NOTE: Ctrl - Shift - F in Eclipse formats the c-code
;; fastest serial speed without probes: 2MHz
;; fastest serial speed with logic analyzer on TX pin: 500kHz
;; minicom -D /dev/ttyACM0 -8 -b 500000