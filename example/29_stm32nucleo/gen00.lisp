(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))
; (setf *features* (union *features* '(:generic-c)))
(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))



(in-package :cl-cpp-generator2)
(setf *auto-keyword* "__auto_type")


(setf *features* (union *features* `(:dac1
				     :adc1
				     :adc2
				     :opamp1)))

(setf *features* (set-difference *features*
				 '(;:dac1
					;:adc1
				   :adc2
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
  (let* ((n-channels 40 ;(* 16 1024)
	  )
	(n-tx-chars (* 4 128))
	(n-dac-vals (- (floor n-channels 2) 0))
	(log-max-entries (* 2 1024))
	(log-max-message-length 27)
	(global-log-message nil))
    (defun global-log (msg) 
	   `(progn
	      (let (;(huart2)
		    (htim5))
		(declare (type "extern UART_HandleTypeDef" huart2)
			 (type "extern TIM_HandleTypeDef" htim5))
		,(let (;(report (format nil "~a\\r\\n" msg))
		       ;(i 0)
		       )
		   `(let (;(c_msg (string ,report))
			  )
		      (declare (type "const char*" c_msg))
	       
		     #+nil  (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*" c_msg ;(string ,report)
							   )
						   ,(+ -2 (length report)))
		     (progn
		       (let ((prim (__get_PRIMASK)))
			 (__disable_irq)
			(do0 ;; https://stm32f4-discovery.net/2015/06/how-to-properly-enabledisable-interrupts-in-arm-cortex-m/
			 (setf (aref glog_ts glog_count)
			       htim5.Instance->CNT)
			 ,(progn
			    (setf global-log-message (append global-log-message (list msg)))
			    ;(defparameter *bla* global-log-message)
			    `(setf (aref glog_msg glog_count)
				   
				   ,(position msg global-log-message :test #'string= )
				   ;,(length global-log-message)
				   ))
			 #+readable_log (let ((p (ref (aref (dot (aref glog glog_count)
						   msg_str) 0))))
			    ,@(loop for e across (subseq msg 0 (min (length msg) (- log-max-message-length 1)))
				 collect
				   (prog1
				       `(do0 (setf (aref p ,i) (aref c_msg ,i) ; (char ,e)
						   )
					     )
				     (incf i)))
			    (setf (aref p ,i) 0))
			 (do0
			  (incf glog_count)
			  (when (<= ,log-max-entries glog_count)
			    (setf glog_count 0))))
			(unless prim
			  (__enable_irq)))))))))
    (progn
      (define-part
	 `(main.c Includes 
		  (include <stdio.h>
			   <math.h>
			   <pb_encode.h>
			   <pb_decode.h>
			   "simple.pb.h"
			   ;<stm32l4xx_hal_tim.h>
			   )))
      (define-part
	  `(main.c PV
		   (do0
		    (do0 

		     (let ((glog_ts)
			   (glog_msg)
			   (glog_count))
		       (declare (type (array uint32_t ,log-max-entries) glog_ts)
				(type (array uint8_t ,log-max-entries) glog_msg)
				(type int glog_count)))

		     #+nil(do0 
		      (defstruct0 log_t
			  (ts uint32_t)
		       
			(msg uint16_t)
			#+readable_log (,(format nil "msg_str[~a]" log-max-message-length) uint8_t)
		       
			)
		      (let (
			    (glog)
			    (glog_count 0)
			    )
			(declare (type (array log_t ,log-max-entries)
					; uint16_t
				       glog)
				 (type int glog_count)
				 ))))
		    (let (#+adc1 (value_adc)
					;#+adc2 (value_adc2) ;; FIXME: 4 byte alignment for dma access
				 #+dac1 (value_dac)
			       
				 (BufferToSend))
		      (declare (type (array  #+adc-interleaved uint8_t
					     #-adc-interleaved uint16_t
					     ,(* #+adc-interleaved 2 n-channels)) value_adc value_adc2)
			       (type (array uint16_t ,n-dac-vals)
					;uint16_t
				     value_dac)
			       (type (array uint8_t ,n-tx-chars) BufferToSend))))))
      (let ((l `((ADC
		  ((ConvHalfCplt :modulo 1 ;1000000
				 )
		   Error
		   (ConvCplt :modulo  1 ; ,(* (/ 1024 128) 30) ;1000000
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
						     ,(global-log (format nil "main.c_0 ~a ~a ~a~@[@~a~]"
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
		    #-nil (do0 (HAL_TIM_Base_Init &htim6)
			       (HAL_TIM_Base_Start &htim6))
		    (do0 (HAL_TIM_Base_Init &htim4)
			 (HAL_TIM_Base_Start &htim4)
			 (HAL_TIM_PWM_Start &htim4 TIM_CHANNEL_1))
		    (do0 (HAL_TIM_Base_Init &htim2)
			 (HAL_TIM_Base_Start &htim2)
			 (HAL_TIM_PWM_Start &htim2 TIM_CHANNEL_1)
			 (HAL_TIM_PWM_Start &htim2 TIM_CHANNEL_2))
		    (do0 (HAL_TIM_Base_Init &htim5) ;; 32bit global time
			 (HAL_TIM_Base_Start &htim5))
		    #+dac1 (do0
			    (dotimes (i ,n-dac-vals)
			      (let ((v (cast uint16_t (rint (* ,(/ 4095s0 2) (+ 1s0 (sinf (* 7 i ,(coerce (/ (* 2 pi) n-dac-vals) 'single-float)))))))
				      ))
				(setf (aref value_dac i) v)))
			    ;(setf (aref value_dac 0) 4095)
			    #+nil (dotimes (i ,(floor n-dac-vals 2))
				    (setf (aref value_dac i) 4095))
			    
			    (HAL_DAC_Init &hdac1)
			    (HAL_DAC_Start &hdac1 DAC_CHANNEL_1)
			    
			    (HAL_DAC_Start_DMA &hdac1 DAC_CHANNEL_1 (cast "uint32_t*" value_dac) ,n-dac-vals
					       DAC_ALIGN_12B_R))
		    #+opamp1 (HAL_OPAMP_Start &hopamp1)

		    #+adc-interleaved (do0
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
		    #+adc-interleaved
		    (do0
		     (HAL_ADCEx_MultiModeStart_DMA &hadc1 (cast "uint32_t*" value_adc) ,n-channels)
		     )
		     (do0
		     #+adc2 (HAL_ADC_Start_DMA &hadc2 (cast "uint32_t*" value_adc2) ,n-channels)
		     #+adc1 (HAL_ADC_Start_DMA &hadc1 (cast "uint32_t*" value_adc) ,n-channels))
 
		    ,(global-log "main.c_2 adc dmas started")
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
				  #+nil (HAL_ADC_Start &hadc1
						 )
				  #+nil ,(global-log "main.c_3 trigger")
				  #+nil ,(let ((report (format nil "trigger\\r\\n" )))
				     `(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,report))
							     ,(+ -2 (length report))))
				  (HAL_Delay 10)
				  
				  (do0
				   (incf htim2.Instance->CCR2)
				   (when (<= ,(- 80  3) htim2.Instance->CCR2) 
				     (setf htim2.Instance->CCR2 2)))

				  #-nil
				  (progn
				    (let ((message SimpleMessage_init_zero)
					  (stream (pb_ostream_from_buffer BufferToSend (sizeof BufferToSend))))
				      (declare (type SimpleMessage message))
				      ,(let ((str "hello"))
					 `(do0
					   (setf message.id #x55555555)
					   (setf message.timestamp htim5.Instance->CNT)
					   (setf message.phase htim2.Instance->CCR2)
					   (dotimes (i ,n-channels)
					     (setf (aref message.samples i) (aref value_adc i)))
					   ;(strcpy message.name (string ,str))
					   ))
				      
				      (let ((status #+nil (pb_encode_ex &stream SimpleMessage_fields &message PB_ENCODE_DELIMITED )
						    ;(pb_encode_delimited &stream SimpleMessage_fields &message)
						    (pb_encode &stream SimpleMessage_fields &message)
						    )
					    (message_length stream.bytes_written))
					(when status
					 (unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*" BufferToSend) message_length))
					   (Error_Handler))))))
				  #+nil
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
							     #+staadc1
							     (1 
								 (aref value_adc 0) :type "%03d"
								 ) 
							     ,@(loop for i below n-channels
								  collect
								    `(,i 
								  (aref value_adc ,i) :type "%4d"
								  ))
							     #+nil (100
								 (aref value_adc 100) :type "%03d"
								 )
							     #+nil (200
								 (aref value_adc 200) :type "%03d"
								 )
							     (ccr2 htim2.Instance->CCR2 :type "%04ld")
							     (tim2 htim2.Instance->CNT :type "%4ld")
							     (tim4 htim4.Instance->CNT :type "%4ld")
							 (tim5 htim5.Instance->CNT :type "%9ld") 
							 (tim6 htim6.Instance->CNT :type "%05ld")
							 (log# glog_count :type "%04d")
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
											    (declare (ignorable v))
											    (format nil "~a"
												    ;name
												    type)))
										      l)))
							     ,@(mapcar #'(lambda (x)
									   (destructuring-bind (name v &key (type "%d")) x
									     (declare (ignorable name v type))
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
      (let ((l `(,@(loop for e in `(USART2 (DMA1_Channel7 :comment USART2_TX)
					   (DMA1_Channel2 :comment TIM2)
					   (DMA1_Channel1 :modulo 1; 1000000
							  :comment ADC1)
					   (DMA1_Channel3 :comment DAC_CH1)
					   ;#+dac1
					   TIM6_DAC
					   (SysTick :modulo  1000
						    ) ;; only show every 1000th interrupt
					   PendSV DebugMonitor SVCall
					   UsageFault BusFault MemoryManagement HardFault
					   NonMaskableInt)
		      collect
			(if (listp e)
			    (destructuring-bind (name &key (modulo 1) (comment nil)) e
				(list (format nil "~a_IRQn 0" name   ;~@[<~a>~] comment
					      )
				      modulo comment))
			    (list (format nil "~a_IRQn 0" e)
				  1)))
		   )))
	(loop for f in l do
	     (destructuring-bind (e modulo &optional (comment "")) f
	      (define-part 
		  `(stm32l4xx_it.c
		    ,e
		    (do0
		     ,(if (eq modulo 1)
			  `(do0
			    ,(global-log
			      (format nil "stm32l4xx_it.c_~a" e))
			    #+nil(let ((huart2))
				   (declare (type "extern UART_HandleTypeDef" huart2))
				   (HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,(format nil "~a\\r\\n" e)))
					      
							  ,(+ 2 (length e)))))
			  `(progn
			     (do0	;let ((huart2))
					;(declare (type "extern UART_HandleTypeDef" huart2))
			
			      (let ((count 0))
				(declare (type "static int" count))
				(incf count)
				(when (== 0 (% count ,modulo))
				  ,(global-log (format nil "stm32l4xx_it.c_~a#~a~@[<~a>~]" e modulo comment))
				  #+nil ,(let ((report (format nil "~a#~a\\r\\n" e modulo)))
					   `(HAL_UART_Transmit_DMA &huart2 (cast "uint8_t*"  (string ,report))
								   ,(+ -2 (length report)))
					   #+nil `(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 (string ,report)
											    ,(length report)))
						    (Error_Handler))))))))
		     ))))))

      (let ((l `(,@(loop for e in `(USART2 #+dac1 DAC1 #+adc1 ADC1
					   #+adc2 ADC2
					   )
		      appending
			(list ;; MSP means mcu support package
			 (format nil "~a_MspInit 1" e)
			 (format nil "~a_MspDeInit 1" e)
			 ))
		   ,@(loop for e in `(TIM2 TIM4 TIM5 TIM6)
		      appending
			(list ;; MSP means mcu support package
			 (format nil "~a_MspInit 1" e)
			 (format nil "~a_MspPostInit 1" e)
			 (format nil "~a_MspDeInit 1" e)
			 ))
		   )))
	(loop for e in l do
	 (define-part 
	     `(stm32l4xx_hal_msp.c
	       ,e
	       (progn
		  ,(global-log (format nil "stm32l4xx_hal_msp.c_~a" e))
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
		 ,(global-log (format nil "main.c_~a" e))
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
		     (let ((glog_ts)
			   (glog_msg)
			   (glog_count))
		       (declare (type (array "extern uint32_t" ,log-max-entries) glog_ts)
				(type (array "extern uint8_t" ,log-max-entries) glog_msg)
				(type "extern int" glog_count)))
					;(include <stm32l4xx_hal_tim.h>)
		     #+nil(do0
		      (defstruct0 log_t
			  (ts uint32_t)
					
			(msg uint16_t)
			#+readable_log (,(format nil "msg_str[~a]" log-max-message-length) uint8_t)
			)
		      
		      
		      (let (
			    (glog)
			    (glog_count)
			    )
			(declare (type (array "extern log_t" ,log-max-entries) glog)
				 (type "extern int" glog_count)
				 ))))
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
	       ))))
    (let* ((pbdir "/home/martin/stage/cl-cpp-generator2/example/29_stm32nucleo/source/")
	  (fn (format nil  "~a/simple.proto" pbdir)))
      (write-source fn
		   `(do0
		     (setf syntax (string "proto2"))
		     "import \"nanopb.proto\";"
		     (space "message SimpleMessage"
			    (progn
			      
			      (setf "required int32 id" 1)
			      ;; If (nanopb).fixed_count is set to true and (nanopb).max_count is also set, the field for the actual number of entries will not by created as the count is always assumed to be max count.

			      (setf "required int32 timestamp" 2)
			      (setf "required int32 phase" 3)
			      (setf "repeated uint32 samples" ,(format nil "4 [packed=true, (nanopb).max_count=~a, (nanopb).fixed_count=true]" n-channels))
			      ;(setf "required string name" "3 [(nanopb).max_size = 40]")
			      
			      ))))
      
      (sb-ext:run-program "/home/martin/src/nanopb/generator/protoc" (list  (format nil "--nanopb_out=~a" pbdir)
									    fn)
			  )
      (sb-ext:run-program "/bin/sh" (list  "/home/martin/stage/cl-cpp-generator2/example/29_stm32nucleo/source/copy_protobuf.sh")
			  ))
     
    
    (with-open-file (s "/home/martin/stage/cl-cpp-generator2/example/29_stm32nucleo/source/messages.txt"
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in global-log-message and i from 0 do
	   
	   (format s "~d ~a~%" i e)))))

;; NOTE: Ctrl - Shift - F in Eclipse formats the c-code
;; fastest serial speed without probes: 2MHz
;; fastest serial speed with logic analyzer on TX pin: 500kHz
;; minicom -D /dev/ttyACM0 -8 -b 500000

#+nil
(position "main.c_3 trigger" *bla* :test #'string= )
