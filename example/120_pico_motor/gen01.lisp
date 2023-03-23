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
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
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
       (return 0)))))

