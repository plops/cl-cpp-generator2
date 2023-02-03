(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "112_usb")
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

  (let* ((notebook-name "perpare_compilation")
	 (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/p~a_~a" *path* *idx* notebook-name)
 
     `(do0
       (do0
	"#!/usr/bin/env python3"
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

	#+nil
	(do0 (setf qtflags_string (subprocess.run
			       (list (string "pkg-config")
				     (string "Qt5Gui")
				     (string "Qt5Widgets")
				     (string "--cflags"))
			       :capture_output True))
	     (setf qtflags (dot qtflags_string
				stdout
				(decode (string "utf-8"))
				(rstrip)
				(split (string " ")))))
	(setf cflags (list ,@(mapcar (lambda (x)
				       `(string ,x))
				     `("-std=c++20" "-ggdb" "-O1" 
						    )))
	      )
	(setf cmd (+ (list ,@(mapcar (lambda (x)
				     `(string ,x))
				   `("clang++"
				     "-###" "main.cpp"
				     "-c" )))
		     cflags
		     ;qtflags
		     ))
	(print (dot (string "calling : subprocess.run {}")
		    (format (dot (string " ")
			  (join cmd)))))
	(setf out
	      (subprocess.run
	       
	       cmd
	       :capture_output True
	       ;:shell True
	       ))

	

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
	#+nil
	(print clang_line1)
	(with (as (open (string "compile01.sh") (string "w"))
		  f)
	      (f.write (+ (string "time ") clang_line1 (string " module.modulemap -o fatheader.pcm -emit-module -fmodules -fmodule-name=fatheader"))))
	(with (as (open (string "compile02.sh") (string "w"))
		  f)
	      (for (file (list (string "main")
			       (string "UsbError")))
		   (setf flags (dot (string " ")
						 (join
						  (+ cflags
					;qtflags
						     ))))
		   (f.write (fstring "time clang++ {flags} -fmodule-file=fatheader.pcm {file}.cpp -c -o {file}.o\\n")
				    ))
	     
	      (f.write (dot (string "time clang++ {} main.o UsbError.o -o main `pkg-config libusb-1.0 --libs`\\n")
			    (format (dot (string " ")
					 (join
					  (+ cflags
					     ;qtflags
					     )))))))
	)
       )
     )))



