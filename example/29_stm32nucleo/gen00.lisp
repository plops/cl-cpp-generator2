(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))
(setf *features* (union *features* '(:generic-c)))
(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `(:dac1
				     :adc1
				     :adc2
				     :opamp1)))

(setf *features* (set-difference *features*
				 '(:dac1
					;:adc1
				   ;:adc2
				   :opamp1
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
  (let ((n-channels (* 2))
	(n-tx-chars 128)
	(n-dac-vals 4096)
	(log-max-entries 99)
	(log-max-message-length 27))
    (defun uartprint (msg)
      `(progn
	 (let ((huart2)
	      (htim2))
	  (declare (type "extern UART_HandleTypeDef" huart2)
		   (type "extern TIM_HandleTypeDef" htim2))
	  ,(let ((report (format nil "~a\\r\\n" msg))
		 (i 0))
	     `(let ((c_msg (string ,report)))
		(declare (type "const char*" c_msg))
	       
		(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*" c_msg ;(string ,report)
						     )
				       ,(+ -2 (length report)))
		(setf (dot (aref glog glog_count)
			   ts)
		      htim2.Instance->CNT
					;(__HAL_TIM_GetCounter htim2)
		      )
		(let ((p (ref (aref (dot (aref glog glog_count)
					 msg) 0))))
		  ,@(loop for e across (subseq msg 0 (min (length msg) (- log-max-message-length 1))) collect
			 (prog1
			     `(do0 (setf (aref p ,i) (aref c_msg ,i) ; (char ,e)
					 )
				   )
			   (incf i)))
		  (setf (aref p ,i) 0))
		(do0
		 (incf glog_count)
		 (when (<= ,log-max-entries glog_count)
		   (setf glog_count 0))))))))
    (progn
      (define-part
	 `(main.c Includes 
		  (include <stdio.h>
			   <math.h>
			   ;<stm32l4xx_hal_tim.h>
			   )))
      (define-part
	  `(main.c PV
		   (do0
		    (do0 
		     
		     (defstruct0 log_t
			 (ts uint32_t)
		       (,(format nil "msg[~a]" log-max-message-length) uint8_t)
		       )
		     (let (
			   (glog)
			   (glog_count 0)
			   )
		       (declare (type (array log_t ,log-max-entries) glog)
				(type int glog_count)
				)))
		    (let (#+adc1 (value_adc)
					;#+adc2 (value_adc2) ;; FIXME: 4 byte alignment for dma access
				 #+dac1 (value_dac)
			       
				 (BufferToSend))
		      (declare (type (array  uint8_t
					;uint16_t
					     ,(* 2 n-channels)) value_adc value_adc2)
			       (type (array uint16_t ,n-dac-vals)
					;uint16_t
				     value_dac)
			       (type (array uint8_t ,n-tx-chars) BufferToSend))))))
      (let ((l `((ADC
		  ((ConvHalfCplt :modulo 1000000)
		   Error
		   (ConvCplt :modulo  ,(* (/ 1024 128) 30) ;1000000
			     )
		   ))
		 (UART (Error TransmitCplt AbortOnError))
		 #+dac1 (DAC (Error ConvCplt ConvHalfCplt) :channels (Ch1 Ch2)))))
	(define-part
	    ;; USE_HAL_UART_REGISTER_CALLBACKS
	    ;; should be  defined to 0 in  stm32l4xx_hal_conf.h but it is not there
	    `(main.c 0
		     (do0
		      ,@(loop for e in l appending
			     (destructuring-bind (module irqs &key (channels `(""))) e
			       (loop for ch in channels append
				    (loop for irq in irqs collect
					 (let ((irq-name irq)
					       (irq-mod 1)) 
					   (when (listp irq)
					      (destructuring-bind (irq-n &key (modulo 1)) irq
						(setf irq-name irq-n
						      irq-mod modulo)))
					  `(defun ,(format nil "HAL_~a_~aCallback~a" module irq-name ch)
						   (arg)
						 (declare (type ,(format nil "~a_HandleTypeDef*" module)
								arg))
						 (let ((output_p 1))
						   (declare (type ,(if (eq 1 irq-mod)
								       "const int"
								       "int") output_p))
						   ,(if (eq irq-mod 1)
							`(comments "no counter")
							`(let ((count 0))
							   (declare (type "static int" count))
							   (incf count)
							   (unless (== 0 (% count ,irq-mod))
							     (setf output_p 0))))
						   (when output_p
						     ,(uartprint (format nil "~a ~a ~a~@[@~a~]"
									 module irq-name ch
									 (unless (eq 1 irq-mod)
									   irq-mod)))
						     #+nil(let ((huart2))
						     (declare (type "extern UART_HandleTypeDef" huart2))
						     ,(let ((report (format nil "~a ~a ~a ~@[@~a~]\\r\\n"
									    module irq-name ch (unless (eq 1 irq-mod)
												 irq-mod))))
							`(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*" (string ,report))
										,(+ -2 (length report)))
							#+nil `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,report)
													 ,(length report)))
								 (Error_Handler))))))))))))))))
      (define-part 
	  `(main.c 2
		   (do0
		    (do0 (HAL_TIM_Base_Init &htim6)
			 (HAL_TIM_Base_Start &htim6))
		    (do0 (HAL_TIM_Base_Init &htim2)
			 (HAL_TIM_Base_Start &htim2))
		    #+dac1 (do0
			    (HAL_DAC_Init &hdac1)
			    
			    (HAL_DAC_Start &hdac1 DAC_CHANNEL_1)
			    (dotimes (i ,n-dac-vals)
			      (let ((v (cast uint16_t (rint (* ,(/ 4095s0 2) (+ 1s0 (sinf (* i ,(coerce (/ (* 2 pi) n-dac-vals) 'single-float)))))))))
			       (setf (aref value_dac i) v)))
			    
			    (HAL_DAC_Start_DMA &hdac1 DAC_CHANNEL_1 (cast "uint32_t*" value_dac) ,n-dac-vals
						     DAC_ALIGN_12B_R))
		    #+opamp1 (HAL_OPAMP_Start &hopamp1)

		    (do0
		     (let ((mode))
		       (declare (type ADC_MultiModeTypeDef mode))
		       (setf mode.Mode ADC_DUALMODE_INTERL  ;; ADC_HAL_EC_MULTI_MODE
			     mode.DMAAccessMode ADC_DMAACCESSMODE_8_6_BITS ;; ADC_HAL_EC_MULTI_DMA_TRANSFER_RESOLUTION
			     mode.TwoSamplingDelay ADC_TWOSAMPLINGDELAY_1CYCLE ;; ADC_HAL_EC_MULTI_TWOSMP_DELAY
			     )
		      (HAL_ADCEx_MultiModeConfigChannel &hadc1 &mode)))
		    
		    #+adc2 (do0 
			    ;(HAL_ADC_Init &hadc2)
					;(HAL_ADC_Start_IT &hadc2)
			    (HAL_ADCEx_Calibration_Start &hadc2 ADC_SINGLE_ENDED)
				
			    )
		    #+adc1 (do0; (HAL_ADC_Init &hadc1)
				;(HAL_ADC_Start_IT &hadc1)

				
				(HAL_ADCEx_Calibration_Start &hadc1 ADC_SINGLE_ENDED)
				
				)
		    (do0
		     (HAL_ADCEx_MultiModeStart_DMA &hadc1 (cast "uint32_t*" value_adc) ,n-channels)
		     )
		    #+nil (do0
		     #+adc2 (HAL_ADC_Start_DMA &hadc2 (cast "uint32_t*" value_adc2) ,n-channels)
		     #+adc1 (HAL_ADC_Start_DMA &hadc1 (cast "uint32_t*" value_adc) ,n-channels))

		    ,(uartprint "adc dmas started")
		    #+nil ,(let ((report (format nil "adc dmas started\\r\\n" )))
				     `(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,report))
							     ,(+ -2 (length report))))
		    )))
      (define-part 
	  `(main.c 3
		   (do0
		    (do0
			    
		     (progn
			      (let ((count))
				(declare (type "static int" count))
				(incf count)
				(when (<= ,(expt 2 12) count)
				  (setf count 0))
				#+nil (setf value_dac ;(aref value_dac count)
					      count
					      )
				  #+nil (if (< value_dac ,(- (expt 2 12) 1))
					    (incf value_dac)
					    (setf value_dac 0))
				  #+nil (HAL_DAC_SetValue &hdac1 DAC_CHANNEL_1 DAC_ALIGN_12B_R value_dac ; (aref value_dac count)
							  )
				  (HAL_ADC_Start &hadc1
						 )
				  ,(uartprint "trigger")
				  #+nil ,(let ((report (format nil "trigger\\r\\n" )))
				     `(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,report))
							     ,(+ -2 (length report))))
				  (HAL_Delay 10)
				  (progn
				    ;; online statistics https://provideyourown.com/2012/statistics-on-the-arduino/
				    (let (;(avg 0s0)
					   ; (var 0s0)
					   ; (std 0s0)
					  )
				      #+nil (do0 (dotimes (i ,n-channels)
					     (incf avg (aref value_adc i)))
					   (setf avg (/ avg ,(* 1s0 n-channels))))
				      #+nil (do0 (dotimes (i ,n-channels)
					     (let ((h (- (aref value_adc i)
							 avg)))
					       (incf var (* h h))))
					   (setf var (/ var ,(* 1s0 n-channels)))
					   (setf std (sqrtf var)))
				      ,(let ((l `(#+srtadac1 (dac (aref value_dac count)
							      )
							 #+adc1 (1 ;USE_HAL_UART_REGISTER_CALLBACKS
								 (aref value_adc 0) :type "%d"
								 )
							 #+danadc2
							 (2 ;USE_HAL_UART_REGISTER_CALLBACKS
							  (aref value_adc2 0) :type "%d"
							  )
							 #+satadc1 (avg ;USE_HAL_UART_REGISTER_CALLBACKS
								    avg :type "%8.0f"
								    )
							 #+satadc1 (std ;USE_HAL_UART_REGISTER_CALLBACKS
								    std :type "%3.1f"
								    ))))
					 `(let ((n (snprintf (cast char* BufferToSend)
							     ,n-tx-chars
							     (string ,(format nil "~{~a~^ ~}\\r\\n"
									      (mapcar #'(lambda (x)
											  (destructuring-bind (name v &key (type "%d")) x
											    (format nil "~a=~a"
												    name type)))
										      l)))
							     ,@(mapcar #'(lambda (x)
									   (destructuring-bind (name v &key (type "%d")) x
									     v))
								       l))))
					    (declare (type int n))
					    (unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*" BufferToSend) n))
					      (Error_Handler)))))))))
		    

		    
		    "}"
		    
		    )))

      (define-part
	  `(stm32l4xx_it.c
	    Includes
	    (do0
	      (include 
	      "global_log.h"
		      ))))
      (define-part
	  `(stm32l4xx_hal_msp.c
	    Includes
	    (do0
	     (include 
	      "global_log.h"))))
      (let ((l `(,@(loop for e in `(USART2 DMA1_Channel7
					   DMA1_Channel2
					   (DMA1_Channel1 :modulo 1000000)
					   DMA1_Channel3
					   ;#+dac1
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
		       ,(uartprint e)
		       #+nil(let ((huart2))
			 (declare (type "extern UART_HandleTypeDef" huart2))
			 (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,(format nil "~a\\r\\n" e)))
					      
					      ,(+ 2 (length e)))))
		     `(progn
			(do0 ;let ((huart2))
			 ;(declare (type "extern UART_HandleTypeDef" huart2))
			
			 (let ((count 0))
			   (declare (type "static int" count))
			   (incf count)
			   (when (== 0 (% count ,modulo))
			     ,(uartprint (format nil "~a#~a" e modulo))
			     #+nil ,(let ((report (format nil "~a#~a\\r\\n" e modulo)))
				`(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,report))
							,(+ -2 (length report)))
				#+nil `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,report)
										 ,(length report)))
					 (Error_Handler))))))))
		)))))

      (let ((l `(,@(loop for e in `(USART2 #+dac1 DAC1 #+adc1 ADC1
					   #+adc2 ADC2)
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
		  ,(uartprint (format nil "~a" e))
		 #+Nil
		 (let ((huart2))
		   (declare (type "extern UART_HandleTypeDef" huart2))
		  ,(let ((report (format nil "~a\\r\\n" e)))
		     `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*" (string ,report))
								,(+ -2 (length report))))
			(Error_Handler))))
		
		 )))))

      (let ((l `(,@(loop for e in `(USART2 #+dac1 DAC1 #+adc1 ADC1
					   #+adc2 ADC2)
		      collect 
			(format nil "~a_Init 0" e)
			 
			 )
		   )))
	(loop for e in l do
	 (define-part 
	     `(main.c
	       ,e
	       (progn
		 ,(uartprint (format nil "~a" e))
		#+nil (let ((huart2)) 
		   (declare (type "extern UART_HandleTypeDef" huart2))
		  ,(let ((report (format nil "~a\\r\\n" e)))
		     `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,report))
								,(+ -2 (length report))))
			(Error_Handler))))
		
		))))))
    (write-source "/home/martin/STM32CubeIDE/workspace_1.4.0/nucleo_l476rg_dual_adc_dac/Core/Src/global_log.h"
		  `(do0
		    (do0 
		     ;(include <stm32l4xx_hal_tim.h>)
		     (defstruct0 log_t
			 (ts uint32_t)
		       (,(format nil "msg[~a]" log-max-message-length) uint8_t)
		       )
		     (let (
			   (glog)
			   (glog_count)
			   )
		       (declare (type (array "extern log_t" ,log-max-entries) glog)
				(type "extern int" glog_count)
				)))
		    ))
    (loop for e in *parts* and i from 0 do
	 (destructuring-bind (&key name file code) e
	   ;; open the file that we will modify
	   (let* ((full-fn (format nil "/home/martin/STM32CubeIDE/workspace_1.4.0/nucleo_l476rg_dual_adc_dac/Core/Src/~a" file))
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
