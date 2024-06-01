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
  (defparameter *source-dir* #P"example/151_raw_ethernet/source03/src/")
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
      cstring
      chrono
      thread
      )

     (comments "Note: not working, yet")

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
	    #+nil (comments "SOCK_STREAM .. merges packets together"
		      "SOCK_DGRAM  .. don't merge packets together"
		      "SOCK_RAW    .. capture ethernet header as well")
	    (let ((sockfd (socket AF_PACKET
				  SOCK_RAW
				  (htons ETH_P_ALL)
				  ))))

	    
	    (do0
	     (comments "bind socket to the hardware interface")
	     (let ((ifindex (static_cast<int> (if_nametoindex (string "wlan0"))))
		   (ll (sockaddr_ll (curly (= .sll_family AF_PACKET)
					   (= .sll_protocol (htons ETH_P_ALL))
					   (= .sll_ifindex ifindex)
					   (= .sll_hatype ARPHRD_ETHER)
					   (= .sll_pkttype (or PACKET_HOST
							       PACKET_OTHERHOST
							       PACKET_BROADCAST))
					;(= .sll_halen 0)
					   ))))
	       ,(lprint :vars `(ifindex))
	       (bind sockfd (reinterpret_cast<sockaddr*> &ll)
		 (sizeof ll))
	       ))

	    (do0
	     (comments "define version")
	     (let ((version TPACKET_V2))
	       (setsockopt sockfd
			   SOL_PACKET
			   PACKET_VERSION
			   &version
			   (sizeof version))))

	    (do0
	     (comments "configure ring buffer")
	     (let ((block_size (static_cast<uint32_t> (* 1 (getpagesize))))
		   (block_nr 2U)
		   (frame_size 128U ;2048U
			       )
		   (frame_nr (* (/ block_size frame_size)
				block_nr))
		   (req (space tpacket_req (curly (= .tp_block_size block_size)
						  (= .tp_block_nr block_nr)
						  (= .tp_frame_size  frame_size)
						  (= .tp_frame_nr frame_nr)
						  ))))

	       ,(lprint :vars `(block_size block_nr frame_size frame_nr))
	       (setsockopt sockfd SOL_PACKET
			   PACKET_RX_RING
			   &req
			   (sizeof req))))

	    (do0
	     (comments "map the ring buffer")
	     (let ((mmap_size (* block_size block_nr))
		   (mmap_base (mmap 0 mmap_size (or PROT_READ PROT_WRITE)
				    MAP_SHARED sockfd 0))))
	     (let ((rx_buffer_size (* block_size block_nr))
		   (rx_buffer_addr mmap_base)
		   ;(rx_buffer_idx 0)
		   (rx_buffer_cnt (/ (* block_size block_nr)
				     frame_size)))
	       ,(lprint :vars `(rx_buffer_size
				rx_buffer_cnt)))
	     )

	    (let ((idx 0)))
	    (while true
		   (let ((pollfds (pollfd (curly (space (= .fd sockfd))
						 (space (= .events POLLIN))
						 (space (= .revents 0)))))
			 (poll_res (ppoll &pollfds 1 nullptr nullptr)))

	      
		     (do0
		
		      (when (& POLLIN pollfds.revents)
			(setf idx 0)
			(let ((base (+ (reinterpret_cast<uint8_t*> rx_buffer_addr)))
			      (header (reinterpret_cast<tpacket2_hdr*> (+ base
								     (* idx frame_size)))))
			  ;(declare (type "volatile tpacket2_hdr*" header))
			  )

			(let ((status (& header->tp_status TP_STATUS_USER))))

			(while status 

			       (do0
		
		  
				(let ((data (+ (reinterpret_cast<uint8_t*> header) header->tp_net))
				      (data_len header->tp_snaplen)
				      (timepoint (+ (std--chrono--system_clock--from_time_t header->tp_sec)
					     (std--chrono--nanoseconds  header->tp_nsec)))
				      (time (std--chrono--system_clock--to_time_t timepoint))
				      (local_time (std--localtime &time ))
				      (local_time_hr (std--put_time local_time (string "%Y-%m-%d %H:%M:%S")))))
				
				,(lprint :vars `(local_time_hr))
				(do0 (do0 (comments "hand frame back to kernel")
					  (setf header->tp_status TP_STATUS_KERNEL))
				     (setf idx (% (+ idx 1)
						  rx_buffer_cnt))
				     (setf  header (reinterpret_cast<tpacket2_hdr*>
						    (+ base
						       (* idx frame_size))))
				     (setf 
				      status (& header->tp_status TP_STATUS_USER)))))))
		     ))


	    
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
