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

  (load "util.lisp")
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
      linux/ip.h
      linux/udp.h
      net/if.h
      netinet/ether.h
      sys/ioctl.h
      sys/socket.h
      unistd.h

      iostream
      array
      span
      string
      system_error
      format

      cstring)
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
	    (comments "Open PF_PACKET socket")
	    (let ((ifName (string "lo"))
		  (sockfd (socket PF_PACKET SOCK_RAW (htons (hex #x800)))))
	      (check sockfd (string "Failed to create socket")))
	    (do0
	     (comments "Set interface to promiscuous mode")
	     (let ((ifopts "ifreq{}"))
	       (strncpy ifopts.ifr_name
			ifName
			(- IFNAMSIZ 1))
	       (check (ioctl sockfd SIOCGIFFLAGS &ifopts)
		      (string "Failed to get interface flags"))
	       (setf ifopts.ifr_flags (logior ifopts.ifr_flags
					     IFF_PROMISC))
	       (check (ioctl sockfd SIOCSIFFLAGS &ifopts)
		      (string "Failed to set interface flags"))))
	    (do0
	     (comments "Allow the socket to be reused")
	     (let ((sockopt SO_REUSEADDR))
	       (check
		(setsockopt sockfd SOL_SOCKET
			    SO_REUSEADDR
			    &sockopt (sizeof sockopt))
		(string "Failed to set socket option SO_REUSEADDR"))))

	    (do0
	     (comments "Bind to device")
	     (check (setsockopt sockfd SOL_SOCKET SO_BINDTODEVICE
				ifName
				(- IFNAMSIZ 1))
		    (string "Failed to set socket option SO_BINDTODEVICE")))
	    (do0
	     (comments "Receive loop")
	     (while true
		    (let ((buf "std::array<uint8_t,1024>{}")
			  (numbytes (recvfrom sockfd
					      (buf.data)
					      (buf.size)
					      0
					      nullptr
					      nullptr)))
		      (check numbytes (string "Failed to receive data"))
		      (let ((eh (reinterpret_cast<ether_header*> (buf.data)))
			    (receivedMac ("std::span<const uint8_t, 6>" eh->ether_dhost 6)))
			;,(lprint :vars `(receivedMac))
			
			)
		      )
	      ))
	    (close sockfd)
	    
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
