(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "150_flatbuffers")
  (defparameter *idx* "02")
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

  (let* ((notebook-name "read_image")
	 (cli-args `(
		     (:short "v" :long "verbose" :help "enable verbose output" :action "store_true" :required nil :nb-init True))))
     (write-source
     (format nil "~a/source01/src/p~a_~a" *path* *idx* notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "sudo emerge -av dev-python/flatbuffers"
		 "flatc --python image.fbs")
       (imports (os
		 time
		 flatbuffers
		 (np numpy)))
       ;(imports ((plt matplotlib.pyplot)))
       "from MyImage.Image import Image"

       #+nil(do0
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
				 (- tz))))))
       (with (as (open (string "image.bin")
		       (string "rb"))
		 f)
	     (setf buf (f.read)))
       (setf image (Image.GetRootAsImage (bytes buf)
					 0))
       (setf img (np.frombuffer (image.DataAsNumpy)
				:dtype np.uint8))
       (setf img (img.reshape (tuple (image.Height)
				     (image.Width))))
       
       ))))

 
 
