(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(progn
  (defparameter *project* "121_nanopb")
  (defparameter *idx* "02")
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

  
  
  (let* ((notebook-name "pb_client")
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
	  (setf s (socket.socket socket.AF_INET
				 socket.SOCK_STREAM))
	  (s.connect (tuple (string "localhost")
			    1234))
	  (setf request (DataRequest :count 123
				     :start_index 12345)
		request_string (request.SerializeToString))
	  ,(lprint :vars `(request_string))
	  (setf opacket (Packet :length (len request_string)
				:payload request_string))
	  (setf opacket_string (opacket.SerializeToString))
	  
	  (s.sendall opacket_string
		     #+nil request_string
		     #+nil (bytes (dot (bytearray request_string)
				       (append 0)))
		     #+nil (+ (struct.pack
			       (string ">I")
			       (len request_string))
			      request_string))
	  (time.sleep .2)
	  (setf data (s.recv 9600))
	  ,(lprint :vars `(data))
	  #+nil (do0 (setf response_length (aref (struct.unpack (string ">I")
						      (aref data (slice "" 4)))
					   0))
	       ,(lprint :vars `(response_length)))
	  (setf response_packet (Packet))
	  (response_packet.ParseFromString data)
	  (setf response (DataResponse))
	  #-nil (response.ParseFromString response_packet.payload)
	  #+nil (response.ParseFromString (aref data (slice 4 "")))
	  ,(lprint :vars `(response))
	  (s.close)
	  
	  )
	 ))))

