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
  (defparameter *source-dir* #P"example/151_raw_ethernet/source02/src/")
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
      ;linux/ip.h
      ;linux/udp.h
      net/if.h
      netinet/ether.h
      ;sys/ioctl.h
      sys/socket.h
      unistd.h
      sys/mman.h
      poll.h
      
      cstring
      iostream
      array
      span
      system_error
      format
      cstdint
      cstring)

     (comments "assume we receive a packet for each line of a video camera"
	       "i didn't add error handling. i suggest strace instead")
     
     (defclass+ VideoLine ()
       "public:"
       "uint16_t width;"
       "uint16_t height;"
       "uint32_t timestamp;"
       "std::array<uint8_t,320> imageData;"
       )
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :vars `(argc (aref argv 0)))
       (handler-case
	   (do0
	    (comments "DGRAM .. don't merge packets together")
	    (let ((sockfd (socket PF_PACKET
				  SOCK_DGRAM
				  (htons ETH_P_ALL)
				  ))))
	    (comments "block release timeout 10ms")
	    (let ((packet_size 512)
		  (frame_size packet_size)
		  (frame_nr 2048)
		  (block_nr 2)
		  (req (space tpacket_req3 (curly (= .tp_block_size (/ (* frame_size
									  frame_nr)
								       block_nr))
						  (= .tp_block_nr block_nr)

						  (= .tp_frame_size  frame_size)
						  (= .tp_frame_nr frame_nr)
						  (= .tp_retire_blk_tov 10)
						  (= .tp_sizeof_priv 0)
						  (= .tp_feature_req_word TP_FT_REQ_FILL_RXHASH)
						  						  
						  ;(= .tp_rx_ring 1)
						  ))))
	      
	      (setsockopt sockfd SOL_PACKET
			  PACKET_RX_RING
			  &req
			  (sizeof req)))
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
