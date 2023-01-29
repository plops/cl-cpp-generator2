(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "109_clang_qt_mod_portable")
  (defparameter *idx* "00")
  (defparameter *path* (format nil "/home/martin/stage/cl-cpp-generator2/example/~a/source00/" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (cmd &optional rest)
    `(when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}" cmd rest))
                   (format  (- (time.time) start_time)
                            ,@rest)))))

  (let* ((notebook-name "get_links")
	 (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/p~a_~a" *path* *idx* notebook-name)
 
     `(do0
       (do0
	(imports (	;os
					;sys
			;time
					;docopt
			;pathlib
					;(np numpy)
					;serial
			;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;   scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
					;re
					;json
					; csv
					;io.StringIO
					;bs4
					;requests

					;(np jax.numpy)
					;(mpf mplfinance)

			;argparse
		  ))
	(subprocess.check_output
	 
	 ,(mapcar (lambda (x)
		   `(string ,x))
		 `("clang++"
		     "-###" "source00/main.cpp"
		     "-c" "-std=c++20" "-ggdb" "-O1"))
	 
	 )
	)
       )
     )))

