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
		  array
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
		  (include data.pb.h)))

       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h"))

       (do0
	(defun read_callback (stream buf count)
	  (declare (type pb_istream_t* stream)
		   (type uint8_t* buf)
		   (type size_t count)
		   (values bool))
	  (let ((fd (reinterpret_cast<intptr_t> stream->state))))
	  (when (== 0 count)
	    (return true))
	  
	  (comments "operation should block until full request is satisfied. may still return less than requested (upon signal, error or disconnect)")
	  (let ((result (recv fd buf count MSG_WAITALL)))
	    ,(lprint :msg "read_callback" :vars `(count result))
	    
	    (dotimes (i count)
	      (fmt--print (string "{:02x} ")
			  (aref buf i))
	      #+nil ,(lprint :msg "r" :vars `(i (aref buf i))))
	    (fmt--print (string "\\n"))
	    (when (== 0 result)	      
	      (comments "EOF")
	      (setf stream->bytes_left 0))
	    (return (== count result))))

	(defun write_callback (stream buf count)
	  (declare (type pb_ostream_t* stream)
		   (type "const pb_byte_t*" buf)
		   (type size_t count)
		   (values bool))
	  (let ((fd (reinterpret_cast<intptr_t> stream->state))))
	  (dotimes (i count)
	    ,(lprint :msg "w" :vars `(i (aref buf i))))
	  (return (== count
		      (send fd buf count 0))))
	(defun pb_istream_from_socket (fd)
	  (declare (type int fd)
		   (values pb_istream_t))
	  (let (
		(stream (pb_istream_t (designated-initializer :callback read_callback
							      :state (reinterpret_cast<void*>
								      (static_cast<intptr_t>
								       fd))
							      :bytes_left SIZE_MAX))))
	    (return stream)))

	(defun pb_ostream_from_socket (fd)
	  (declare (type int fd)
		   (values pb_ostream_t))
	  (let (
		(stream (pb_ostream_t (designated-initializer :callback write_callback
							      :state (reinterpret_cast<void*>
								      (static_cast<intptr_t>
								       fd))
							      :max_size SIZE_MAX
							      :bytes_written 0))))
	    (return stream)))
	
	(defun handle_connection (connfd)
	  (declare (type int connfd))
	  (let ((input (pb_istream_from_socket connfd))
		(request (DataRequest)))
	    (unless (pb_decode &input DataRequest_fields &request) 
	      ,(lprint :msg "error decode request"))
	    
	    
	    ,(lprint :msg "request"
		     :vars `(request.count
			     request.start_index))
	    (let ((response (DataResponse))
		  )
	      (setf ,@(loop for e in `((:name index :value 0)
				  (:name datetime :value 123)
				  (:name pressure :value 1234.5)
				  (:name humidity :value 47.3)
				  (:name temperature :value 23.2)
				  (:name co2_concentration :value 456.0))
			    appending
			    (destructuring-bind (&key name value) e
			      `((dot response ,name)
				,value))))
	      
	      (let ((output (pb_ostream_from_socket connfd)))
		(unless 
		    (pb_encode &output DataResponse_fields &response)
		  ,(lprint :msg "error encoding response")))
	      )))
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



