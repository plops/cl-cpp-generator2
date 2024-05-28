(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/151_raw_ethernet/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (comments "based on https://gist.github.com/austinmarton/2862515")
     (include<>
      arpa/inet.h
      linux/if_packet.h
      ;linux/ip.h
      ;linux/udp.h
      net/if.h
      netinet/ether.h
      sys/ioctl.h
      sys/socket.h
      unistd.h

      iostream
      span
      string
      system_error)
     (defun check (result msg)
       (declare (type int result)
		(type "const std::string&" msg))
       (when (== -1 result)
	 (throw (std--system_error errno
				   (std--generic_category)
				   msg))))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (handler-case
	   (do0
	    (let ((ifName (string "eth0"))
		  (sockfd (socket PF_PACKET SOCK_RAW (htons (hex #x800)))))
	      (check sockfd (string "Failed  to create socket"))
	      )
	    )
	 ("const std::system_error&" (ex)
	   (<< std--cerr
	       (string "Error: ")
	       (ex.what)
	       (string " (")
	       (ex.code)
	       (string ")\\n"))
	   (return 1)))
       
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))
