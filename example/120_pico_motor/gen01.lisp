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
     (include "pico/stdlib.h")
     (space enum
	    "{"
	    (comma
	     (= GPIO_OFF 0)
	     (= GPIO_ON 1))
	    "}")
     (space enum
	    "{"
	    (comma
	     (= LED_PIN 25))
	    "}")
     
     (defun main ()
       (declare (values int))

       (stdio_init_all)
       (do0 (gpio_init LED_PIN)
	    (gpio_set_dir LED_PIN GPIO_OUT))
       
       ;(setup_default_uart)
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
		 `(gpio_gpio_init pio (+ pin ,i)))
	 (pio_sm_set_consecutive_pindirs
	  pio sm pin 4 true)
	 (pio_sm_offset &c))
       
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
   :format nil :tidy nil))

