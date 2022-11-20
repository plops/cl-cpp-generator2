(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "86_libtorch")
  (defparameter *idx* "01")
  (defparameter *path* (format nil "/home/martin/stage/cl-cpp-generator2/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun init-lprint ()
    `(def lprint (msg args)
       (when args.verbose
	 (print (dot (string "{} {}")
                     (format  (- (time.time) start_time)
                              msg))))))
  (defun lprint (&key msg vars)
    `(lprint (dot (string ,(format nil "~@[~a ~]~{~a={}~^ ~}" msg (mapcar (lambda (x) (emit-py :code x))
									  vars)))
                  (format ,@vars))
	     args))

  (let* ((notebook-name "view")
	 (cli-args `(
		     (:short "v" :long "verbose" :help "enable verbose output" :action "store_true" :required nil :nb-init True))))
    (write-notebook
     :nb-file (format nil "~a/source/~a_~a.ipynb" *path* *idx* notebook-name)
     :nb-code
     `((python
	(export
	 ,(format nil "#|default_exp p~a_~a" *idx* notebook-name)))
       (python (export
		(do0
					;(comments "this file is based on ")
					;"%matplotlib notebook"
		 (do0

		       (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		       (imports ((plt matplotlib.pyplot)
					;  (animation matplotlib.animation)
					;(xrp xarray.plot)
				 ))

		       (plt.ion)
					;(plt.ioff)
		       ;;(setf font (dict ((string size) (string 6))))
		       ;; (matplotlib.rc (string "font") **font)
		       )
		 (imports (	os
					;sys
				time
					;docopt
				pathlib
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
					;selenium.webdriver ;.FirefoxOptions
				tqdm
				argparse
				torch
				))
		 "import torch.nn.functional as F"
		 (imports-from (torch tensor))



		 #-nil
		 (imports-from  (matplotlib.pyplot
				 plot imshow tight_layout xlabel ylabel
				 title subplot subplot2grid grid text
				 legend figure gcf xlim ylim)
				)

		 (imports-from  (torch
				 linspace
				 randn
				 randint
				 tanh
				 )
				)

		 )
		))

       (python
	(do0
	 (class Args ()
		(def __init__ (self)
		  ,@(loop for e in cli-args
			  collect
			  (destructuring-bind (&key short long help required action nb-init) e
			    `(setf (dot self ,long) ,nb-init)))))
	 (setf args (Args))))
       (python
	(export
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

	 (setf start_time (time.time)
	       debug True)
	 ,(init-lprint)))




       (python
	(export
	 (do0 (setf parser (argparse.ArgumentParser))
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key short long help required action nb-init) e
			`(parser.add_argument
			  (string ,(format nil "-~a" short))
			  (string ,(format nil "--~a" long))
			  :help (string ,help)
					;:required
			  #+nil
			  (string ,(if required
				       "True"
				       "False"))
			  :action ,(if action
				       `(string ,action)
				       "None"))))

	      (setf args (parser.parse_args)))))

       (python
	(export
	 (setf module (torch.jit.load (string "/home/martin/stage/cl-cpp-generator2/example/86_libtorch/source/../source00/b/cp0/dcgan-sample-104.pt")))
	 (setf images (aref ("list" (module.parameters))
			    0))))))))

