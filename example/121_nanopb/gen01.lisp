(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)


(progn
  #+nil
  (progn
    (defparameter *source-dir*       "/home/martin/src/my_fancy_app_name/main/")
    (defparameter *full-source-dir*  "/home/martin/src/my_fancy_app_name/main/"))
  #-nil
  (progn
    (defparameter *source-dir* #P"example/121_nanopb/source01/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (let ((n-fifo (floor 320 1))
	(l-data
	  `((:name temperature :hue 150 ; 
		   
		   :short-name T :unit "°C" :fmt "{:2.2f}")
	    (:name humidity :hue 80	; green
	     :short-name H :unit "%" :fmt "{:2.1f}"
	     )
	    (:name pressure :hue 240	;red
	     :short-name p :unit "mbar" :scale 1s-2 :fmt "{:4.2f}"
	     )
					;(:name gas_resistance :hue 100)
	    )))
    (let ((name `TcpServer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  )
       :implementation-preamble
       `(do0
	 #+nil (space "extern \"C\" "
		(progn
		  ))
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h")))
       :code `(do0
	       (defclass ,name ()	 
		 "public:"
		 (defmethod ,name ()
		   (declare
		    (construct
		     )
		    (explicit)	    
		    (values :constructor))

		   )))))
    
    
    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0
       
       (include<> deque
		;  random
		 ; vector
		 ; algorithm
		  ;cmath
		  )

       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h"))

       
))))



