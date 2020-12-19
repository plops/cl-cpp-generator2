(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)

;; python3 -m torch.utils.bottleneck run_00_start.py

(progn
  (defparameter *path* "/home/martin/stage/cl-cpp-generator2/example/57_torch")
  (defparameter *code-file* "run_01_plot")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
   (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

   (defun lprint (rest)
     `(do0
       (setf frameinfo (inspect.getframeinfo (inspect.currentframe)))
       (print (dot (string3
		    ,(format nil "{:09.5f} {}:{} ~{~a={}~^ ~}"
			     
			     (mapcar #'(lambda (x)
					 (emit-py :code x))
				     rest)))
		   (format (- (time.time) start_time) frameinfo.filename frameinfo.lineno ,@rest)))))
     
  (let* (
	 
	 (code
	  `(do0
	    (do0 "# %% imports"
		 (do0 (imports ((plt matplotlib.pyplot)))
		      (plt.ioff))
		 (imports (		;os
			   ;sys
			   time
					;docopt
			   pathlib
			   (np numpy)
			   ;serial
			   (pd pandas)
			   ;(xr xarray)
			   ;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
			   ;scipy.ndimage
			   ;scipy.optimize
					;nfft
			   ;sklearn
			   ;sklearn.linear_model
			   ;itertools
			   ;datetime
			   ;dask.distributed
					;(da dask.array)
					;PIL
					;libtiff
					;visdom
			   inspect
			   torch
			   (nn torch.nn)
			   (F torch.nn.functional)
			   (optim torch.optim)
			   torchvision
			   (transforms torchvision.transforms)))
		 (setf
	       _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

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
		 (setf start_time (time.time))
		 (setf fns ("list" (dot (pathlib.Path (string "b/"))
					(glob (string "*sample*.pt")))))
		 ,(lprint `(fns))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))

