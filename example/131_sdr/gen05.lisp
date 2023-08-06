(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "131_sdr")
  (defparameter *idx* "05")
  (defparameter *path* (format nil "/home/martin/stage/cl-cpp-generator2/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    msg
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    vars)))
                   (format  (- (time.time) start_time)
                            ,@vars)))))

  
  (defun doc (def)
    `(do0
      ,@(loop for e in def
	      collect
	      (destructuring-bind (&key name val (unit "-") (help name)) e
		`(do0
		  (comments ,(format nil "~a (~a)" help unit))
		  (setf ,name ,val))))))
  
  (let* ((notebook-name "segment")
	 #+nil (cli-args `(
			   (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source05/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (do0
	)
       (do0
	
	#+nil
	(do0
	 
	 (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
	 (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation)
					;(xrp xarray.plot)
		   ))

					;(plt.ion)
	 (plt.ioff)
	 ;;(setf font (dict ((string size) (string 6))))
	 ;; (matplotlib.rc (string "font") **font)
	 )
	#+nil 
	(imports-from  (matplotlib.pyplot
			plot imshow tight_layout xlabel ylabel
			title subplot subplot2grid grid text
			legend figure gcf xlim ylim)
		       )
	(imports (			;	os
					;sys
		  time
					;docopt
					;pathlib
					(np numpy)
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
					;(np numpy)
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
					
					;math
			
					;(np jax.numpy)
					;(mpf mplfinance)
					;(fft scipy.fftpack)
					;argparse
					;torch
					;(mp mediapipe)
					;mss
					;(cv cv2)
		  
		  )))
       
       (setf start_time (time.time)
	     debug True)
       (setf
	_code_git_version
	(string ,(let ((str (with-output-to-string (s)
			      (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
		   (subseq str 0 (1- (length str)))))
	_code_repository (string ,(format nil
					  "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/"
					  *project*))
	_code_generation_time
	(string ,(multiple-value-bind
		       (second minute hour date month year day-of-week dst-p tz)
		     (get-decoded-time)
		   (declare (ignorable dst-p))
		   (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			   hour
			   minute
			   second
			   (nth day-of-week *day-names*)
			   year
			   month
			   date
			   (- tz)))))

       (do0
	(setf dt (np.dtype (list (tuple (string "real")
					np.int16)
				 (tuple (string "imag")
					np.int16)))
	      data_raw (np.fromfile (string "/mnt5/gps/out_pnr4_3_31_202308061610.dat")
				:dtype dt
				:count (// (* 10 1024 1024)
					   4)))
	(setf data (+ (aref data_raw (string "real"))
		      (* 1j (aref data_raw (string "imag"))))))

       
       ))))

