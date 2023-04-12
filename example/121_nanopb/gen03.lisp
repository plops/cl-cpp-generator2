(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "121_nanopb")
  (defparameter *idx* "03")
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

  
  
  (let* ((notebook-name "pb_server")
	 #+nil (cli-args `(
		     (:short "-v" :long "--verbose" :help "enable verbose output" :action "store_true" :required nil))))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* *idx* notebook-name)
     `(do0
       (do0
	,(format nil "#|default_exp p~a_~a" *idx* notebook-name))
	 (do0
	  (comment "pip3 install --user protobuf"
		   "cd /home/martin/stage/cl-cpp-generator2/example/121_nanopb/source01; protoc --proto_path=. --python_out=. data.proto")
	  (imports (;	os
					;sys
		    time
		    socket
		    struct
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
					;  (np numpy)
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
					;mss
			
					;(np jax.numpy)
					;(mpf mplfinance)

					;argparse
					;torch
					
			)))

	 (imports-from  (data_pb2 Packet DataRequest DataResponse))
	 
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
	  (def listen ()
	    (with (as (socket.socket socket.AF_INET
				     socket.SOCK_STREAM)
		      s)
		  (s.bind (tuple (string "localhost")
				 1234))
		  (s.listen)
		  ,(lprint :msg "listening on localhost:1234")
		  (while True
			 (setf (ntuple conn addr) (s.accept))
			 (with conn
			       ,(lprint :msg "connection" :vars `(addr))
			       ,(lprint :msg "wait for DataResponse message")
			       (setf data (conn.recv 1024))
			       (setf buf data)
			       
			       (while data
				      (setf data (conn.recv 1024))
				      (incf buf data))
			       ,(lprint :msg "finished reading" :vars `(buf))
			       (setf imsg (DataResponse))
			       
			       ,(lprint :vars `((dot imsg
				    (ParseFromString buf))))
			       ,(lprint :vars `(,@(loop for e in `(index datetime pressure humidity temperature
									 co2_concentration)
							collect
							`(dot imsg ,e))))
			       ,(lprint :msg "send DataRequest message")
			       (setf omsg (dot (DataRequest :start_index 123
							    :count 42) 
					       (SerializeToString)))
			       (conn.sendall omsg)
			       ,(lprint :msg "connection closed"))))
	    )
	  (listen)
	  
	  )
	 ))))

