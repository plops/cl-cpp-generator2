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
		  subprocess
					;(np jax.numpy)
					;(mpf mplfinance)

			;argparse
		  ))

	(setf cmd (list ,@(mapcar (lambda (x)
				    `(string ,x))
				  `("clang++"
				    "-###" "main.cpp"
				    "-c" "-std=c++20" "-ggdb" "-O1"))))
	(print (dot (string "calling: {}")
		    (format (dot (string " ")
			  (join cmd)))))
	(setf out
	      (subprocess.run
	       
	       cmd
	       :capture_output True))


	;; i am only interested in line after (in-process):
	
	;; clang version 15.0.7 (Fedora 15.0.7-1.fc37)
	;; Target: x86_64-redhat-linux-gnu
	;; Thread model: posix
	;; InstalledDir: /usr/bin
	;;  (in-process)
	;;  "/usr/bin/clang-15" "-cc1" "-triple"
 
	
	(setf count 0
	      start -1
	      clang_line0 None)
	(for (line (dot out
			stderr
			(decode (string "utf-8"))
			(split (string "\\n"))))
	     (when (in (string "(in-process)")
		       line )
	       (setf start count))
	     (when (and (< 0 start) (== (+ 1 start) count))
	       (setf clang_line0 line)
	       #+nil
	       (print (dot (string "{}: '{}'")
			   (format count line))))
	     (incf count))

	(setf clang_line1
	      (dot clang_line0
		   (replace (string "\\\"-o\\\" \\\"main.o\\\"")
			    (string ""))
		   (replace (string "\\\"-main-file-name\\\" \\\"main.cpp\\\"")
			    (string ""))
		   (replace (string "\\\"main.cpp\\\"")
			    (string ""))))
	(print clang_line1)
	
	)
       )
     )))



