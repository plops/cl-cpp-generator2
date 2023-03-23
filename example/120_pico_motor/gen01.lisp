(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/120_pico_motor/source01/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  #+nil (load "util.lisp")

  
  (write-source
   (merge-pathnames #P"main.cpp"
		    *full-source-dir*)
   `(do0
     (include<> iostream)
     (include "motor_library.hpp")
     (space enum
	    "{"
	    (comma
	     (= GPIO_OFF 0)
	     (= GPIO_ON 1))
	    "}")
     (space enum
	    "{"
	    (comma
	     (= MOTOR1_IN1 2)  ;; base pin for motor
	     (= LED_PIN 25)) ;; green led
	    "}")
     "volatile int state1 = 0;"

     (defun pio0_interrupt_handler ()
       (pio_interrupt_clear pio_0 1)
       (if (== 0 state1)
	   (do0
	    (SET_DIRECTION_MOTOR_1 COUNTERCLOCKWISE)
	    (setf state1 1))
	   (if (== 1 state1)
	       (do0
		(SET_DIRECTION_MOTOR_1 CLOCKWISE)
		(setf state1 2))
	       (do0
		(SET_DIRECTION_MOTOR_1 STOPPED)
		(setf state1 0)))
	   )
       (MOVE_STEPS_MOTOR_1 1024))
     
     (defun main ()
       (declare (values int))

       (stdio_init_all)

       (setupMotor1 MOTOR1_IN1 pio0_interrupt_handler)
       
       (do0 (gpio_init LED_PIN)
	    (gpio_set_dir LED_PIN GPIO_OUT))

       (pio0_interrupt_handler)
       (while true
	      (do0
	       (gpio_put LED_PIN GPIO_ON)
	       (sleep_ms 200)
	       (gpio_put LED_PIN GPIO_OFF)
	       (sleep_ms 200)
	       )
	      (<< std--cout
		  (string "hello world")
		  std--endl))
       (return 0))))

  (write-source
   (merge-pathnames #P"motor_library.hpp"
		    *full-source-dir*)
   `(do0
     (include<> iostream)
     ,@(loop for e in `((pico stdlib)
			(hardware pio dma irq)
			)
	     appending
	     (destructuring-bind (pre &rest rest) e
	       (loop for r in rest
		     collect
		     `(include ,(format nil "~a/~a.h" pre r)))))
     (include "stepper.pio.h"
	      "counter.pio.h")

     (space enum
	    "{"
	    (comma
	     (= STOPPED 0)
	     (= CLOCKWISE 1)
	     (= COUNTERCLOCKWISE 2))
	    "}")

     
     ,(let ((half-steps `((1 0 0 1)
			 (1 0 0 0)
			 (1 1 0 0)
			 (0 1 0 0)
			 (0 1 1 0)
			 (0 0 1 0)
			 (0 0 1 1)
			 (0 0 0 1))))
	`(do0
	  (= "unsigned char pulse_sequence_forward[8]"
	     (curly
	      ,@(loop for e in half-steps
		      collect
		      (format nil "0b~{~a~}" e))))
	  (= "unsigned char pulse_sequence_backward[8]"
	     (curly
	      ,@(loop for e in (reverse half-steps)
		      collect
		      (format nil "0b~{~a~}" e))))
	  (= "unsigned char pulse_sequence_stationary[8]"
	     (curly
	      ,@(loop for e in (reverse half-steps)
		      collect
		      0)))))
     "unsigned int pulse_count_motor1 = 1024;"
     "unsigned char* address_pointer_motor1 = pulse_sequence_forward;"
     "unsigned int* pulse_count_motor1_address_pointer = &pulse_count_motor1;"
     "#define MOVE_STEPS_MOTOR_1(a) pulse_count_motor1=a; dma_channel_start(dma_chan_2)"
     "#define SET_DIRECTION_MOTOR_1(a) address_pointer_motor1 = (a==2) ? pulse_sequence_forward : (a==1) ? pulse_sequence_backward : pulse_sequence_stationary"
     "PIO pio_0 = pio0;"
     "int pulse_sm_0 = 0;"
     "int count_sm_0 = 1;"
     (comments "dma channels"
	       "0 .. pulse train to motor1"
	       "1 .. reconfigures and restarts irq 0"
	       "2 .. sends step count to motor 1")
     "int dma_chan_0 = 0;"
     "int dma_chan_1 = 1;"
     "int dma_chan_2 = 2;"
     (defun setupMotor1 (in1 handler)
       (declare (type "unsigned int" in1)
		(type irq_handler_t handler))
       ;; load pio programs into pio0
       (let ((pio0_offset_0 (pio_add_program pio0 &stepper_program))
	     (pio0_offset_1 (pio_add_program pio0 &counter_program)))

	 ;; initialize pio programs
	 (stepper_program_init pio_0 pulse_sm_0 pio0_offset_0 in1)
	 (counter_program_init pio_0 count_sm_0 pio0_offset_1)
	 ;; start pio programs
	 (pio_sm_set_enabled pio_0 pulse_sm_0 true)
	 (pio_sm_set_enabled pio_0 count_sm_0 true)
	 ;; setup interrupts
	 (pio_interrupt_clear pio_0 1)
	 (pio_set_irq0_source_enabled pio_0 PIO_INTR_SM1_LSB true)
	 (irq_set_exclusive_handler PIO0_IRQ_0 handler)
	 (irq_set_enabled PIO0_IRQ_0 true)

	 ;; dma data channels
	 ;; channel zero sends pulse train data to pio stepper machine
	 (let ((c0 (dma_channel_get_default_config dma_chan_0)))
	   ;; 32bit transfers
	   (channel_config_set_transfer_data_size &c0 DMA_SIZE_8)
	   
	   (channel_config_set_read_increment &c0 true)
	   (channel_config_set_write_increment &c0 false)
	   (channel_config_set_dreq &c0 DREQ_PIO0_TX0)
	   (channel_config_set_chain_to &c0 dma_chan_1)

	   (dma_channel_configure
	    dma_chan_0 ;; channel to be configured
	    &c0	       ;; configuration we just created 
	    (-> pio_0  ;; write address (stepper pio tx fifo
		(aref txf pulse_sm_0))
	    address_pointer_motor1
	    8	  ;; number of transfers, each is 4 bytes
	    false ;; don't start immediatly
	    )
	   
	   )

	 ;; channel 1

	 (let ((c1 (dma_channel_get_default_config dma_chan_1)))
	   ;; 32bit transfers
	   (channel_config_set_transfer_data_size &c1 DMA_SIZE_32)
	   
	   (channel_config_set_read_increment &c1 false)
	   (channel_config_set_write_increment &c1 false)
	   ;; chain to other channel
	   (channel_config_set_chain_to &c1 dma_chan_0)

	   (dma_channel_configure
	    dma_chan_1 ;; channel to be configured
	    &c1	       ;; configuration we just created 
	    (-> dma_hw ;; write address: channel 0 read address
		(dot (aref ch dma_chan_0)
		     read_addr))
	    address_pointer_motor1 ;; read address
	    1 
	    false ;; don't start immediatly
	    )
	   
	   )

	 ;; channel 2 (restarts)
	 (let ((c2 (dma_channel_get_default_config dma_chan_2)))
	   ;; 32bit transfers
	   (channel_config_set_transfer_data_size &c2 DMA_SIZE_32)
	   
	   (channel_config_set_read_increment &c2 false)
	   (channel_config_set_write_increment &c2 false)
	  

	   (dma_channel_configure
	    dma_chan_2 ;; channel to be configured
	    &c2	       ;; configuration we just created 
	    (-> pio_0 ;; write to pacer pio tx fifo
		(aref txf count_sm_0))
	    pulse_count_motor1_address_pointer ;; read address
	    1 
	    false ;; don't start immediatly
	    )
	   
	   )
	 ;; start data channels
	 (dma_start_channel_mask (<< 1u dma_chan_0))
	 ))
    ))
  (write-source
   (merge-pathnames #P"stepper.pio"
		    *full-source-dir*)
   `(do0

     (lines ".program stepper"

	    "wait 1 irq 3"
	    "out pins, 4"
	    
	    "% c-sdk {")
     (defun stepper_program_init (pio sm offset pin)
       (declare (type PIO pio)
		(type uint sm offset pin)
		(values "static inline void"))
       (let ((c (stepper_program_get_default_config offset)))
	 (sm_config_set_out_pins &c pin 4)
	 (comments "max clock divider is 65536, a one cycle delay in assembler makes motors slow enough")
	 (sm_config_set_clkdiv &c 65000)
	 (comments "setup autopull, 32bit threshold, right-shift osr")
	 (sm_config_set_out_shift &c 1 1 4)
	 ,@(loop for i below 4 collect
		 `(pio_gpio_init pio (+ pin ,i)))
	 (pio_sm_set_consecutive_pindirs
	  pio sm pin 4 true)
	 (pio_sm_init pio sm offset &c))
       
       )

     "%}")
   :format nil :tidy nil)

  (write-source
   (merge-pathnames #P"counter.pio"
		    *full-source-dir*)
   `(do0

     (lines ".program counter"

	    "pull block"
	    "mov x, osr"

	    "countloop:"
	    "jmp !x done"
	    "irq wait 3"
	    "jmp x-- countloop"

	    "done:"
	    "irq wait 1"
	    
	    "% c-sdk {")
     (defun counter_program_init (pio sm offset)
       (declare (type PIO pio)
		(type uint sm offset)
		(values "static inline void"))
       (let ((c (counter_program_get_default_config offset)))
	 (pio_sm_init pio sm offset &c))
       
       )

     "%}")
   :format nil :tidy nil)

  #+nil
  (write-source
   (merge-pathnames #P"pacer.pio"
		    *full-source-dir*)
   `(do0

     (lines ".program pacer"

	    ;; shift value from osr to scratch x (using autopull)
	    "out x, 32"
	    "countloop:"
	    ;; loop until x hits 0
	    "jmp x-- countloop"

	    ;; wiat for signal to pulse from counter state machine
	    "wait 1 irq 2"
	    ;; signal to send pulse

	    "irq 3"
	    
	    "% c-sdk {")
     (defun pacer_program_init (pio sm offset)
       (declare (type PIO pio)
		(type uint sm offset)
		(values "static inline void"))
       (let ((c (pacer_program_get_default_config offset)))
	 (sm_config_set_out_shift &c 1 1 32)
	 (sm_config_set_clkdiv &c 2)
	 (pio_sm_init pio sm offset &c))
       
       )

     "%}")
   :format nil :tidy nil))

