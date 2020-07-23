(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)




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
	(n-tx-chars 128))
    #+nil (define-part
	 `(main.c 0
		  (let ((value_adc)
			(value_dac)
			(BufferToSend))
		    (declare (type (array uint16_t ,n-channels) value_adc)
			     (type uint16_t value_dac)
			     (type (array uint8_t ,n-tx-chars) BufferToSend)))))
    (progn
      (define-part
	 `(main.c Includes
		  (include <stdio.h>)))
      (define-part
	 `(main.c PV
		  (let ((value_adc)
			(value_dac)
			(BufferToSend))
		    (declare (type (array uint16_t ,n-channels) value_adc)
			     (type uint16_t value_dac)
			     (type (array uint8_t ,n-tx-chars) BufferToSend)))))
      (define-part 
	  `(main.c 2
		   (do0
		    (HAL_DAC_Start &hdac1 DAC_CHANNEL_1)
		    (HAL_ADCEx_Calibration_Start &hadc1 ADC_SINGLE_ENDED)
		    (HAL_ADC_Start_DMA &hadc1 (cast "uint32_t*" value_adc) ,n-channels))))
      (define-part 
	  `(main.c 3
		   (do0
		    (HAL_DAC_SetValue &hdac1 DAC_CHANNEL_1 DAC_ALIGN_12B_R value_dac)
		    (if (< value_dac 2047)
			(incf value_dac)
			(setf value_dac 0))
		    (HAL_Delay 10)

		    (progn
		      (let ((n (snprintf (cast int8_t* BufferToSend)
					 ,n-tx-chars
					 (string "dac=%d adc=%d")
					 value_adc
					 (aref value_adc 0))))
			(declare (type int n))
			(unless (== HAL_OK (HAL_UART_Transmit_DMA &huart2 BufferToSend n))
			  (Error_Handler))))
		    
		    ))))
    
    (loop for e in *parts* and i from 0 do
	 (destructuring-bind (&key name file code) e
	   ;; open the file that we will modify
	   (let* ((full-fn (format nil "/home/martin/STM32CubeIDE/workspace_1.3.0/nucleo_l476rg_dac_adc_loopback2/Core/Src/~a" file))
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
						 (format nil "~a~%~a~%~a~%" start-comment
							 " "
					; (emit-c :code code)
							 end-comment
							 ))))
	       (with-open-file (s full-fn ;"/dev/shm/o.c"
				  :direction :output :if-exists :supersede :if-does-not-exist :create)
		 (write-sequence new s))
	       ))))))
 
