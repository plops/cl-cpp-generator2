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
       (space "extern \"C\" "
		(progn
		  (include<>
		   sys/socket.h
		  sys/types.h
		  netinet/in.h
		  unistd.h
		  pb.h
		  pb_encode.h
		  pb_decode.h)
		  (include data.pb.h
			   
			   )

		  
		  ))

       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h"))

      

       (do0
	
	(defun pb_istream_from_socket (fd)
	  (declare (type int fd)
		   (values pb_istream_t))
	  (let ((stream (pb_istream_t)))
	    (return stream)))
	
	(defun handle_connection (connfd)
	  (declare (type int connfd))
	  (let ((input (pb_istream_from_socket connfd))
		(request (DataRequest)))
	    (unless (pb_decode_delimited &input
					 DataRequest_fields
					 &request))))
	(defun main (argc argv)
	  (declare (values int)
		   (type int argc)
		   (type char** argv))
	  ,(lprint :msg (multiple-value-bind
			      (second minute hour date month year day-of-week dst-p tz)
			    (get-decoded-time)
			  (declare (ignorable dst-p))
			  (format nil "generation date ~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				  hour
				  minute
				  second
				  (nth day-of-week *day-names*)
				  year
				  month
				  date
				  (- tz))))

	  (do0
	   (let ((listenfd (socket AF_INET SOCK_STREAM 0))
		 (reuse (int 1)))
	     (setsockopt listenfd SOL_SOCKET
			 SO_REUSEADDR
			 &reuse
			 (sizeof reuse))
	     (let ((servaddr (sockaddr_in)))
	       (memset &servaddr 0 (sizeof servaddr))
	       ,@(loop for (e f) in `((sin_family AF_INET)
				      (sin_addr.s_addr (htonl INADDR_LOOPBACK))
				      (sin_port (htons 1234)))
		       collect
		       `(setf (dot servaddr ,e )
			      ,f)))
	     (unless (== 0 (bind listenfd
			     ("reinterpret_cast<sockaddr*>"
			      &servaddr)
			     (sizeof servaddr)))
	       ,(lprint :msg "error bind"))
	     (unless (== 0 (listen listenfd 5))
	       ,(lprint :msg "error listen"))
	     (while true
		    (let ((connfd (accept listenfd nullptr nullptr)))
		      (when (< connfd 0)
			,(lprint :msg "error accept"))
		      (handle_connection connfd)
		      (close connfd)
		      ))))
	  (return 0)))

       
))))



