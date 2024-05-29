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
	    (comments "DGRAM .. don't merge packets together")
	    (let ((sockfd (socket PF_PACKET
				  SOCK_DGRAM
				  (htons ETH_P_ALL)
				  ))))
	    (comments "block release timeout 10ms")
	    (let ((packet_size 512U)
		  (frame_size packet_size)
		  (frame_nr 2048U)
		  (block_nr 2U)
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
			  (sizeof req))

	      (let ((ll (sockaddr_ll (curly (= .sll_family AF_PACKET)
					    (= .sll_protocol (htons ETH_P_ALL))
					    (= .sll_ifindex (static_cast<int> (if_nametoindex (string "lo"))))
					    (= .sll_hatype ARPHRD_ETHER)
					    (= .sll_pkttype PACKET_BROADCAST)
					    (= .sll_halen 0)
					    ))))
		(bind sockfd (reinterpret_cast<sockaddr*> &ll)
		  (sizeof ll))
		)
	      (let ((ring_info (tpacket_req3))
		    (len (static_cast<socklen_t> (sizeof ring_info))))
		(getsockopt sockfd SOL_PACKET PACKET_RX_RING &ring_info &len))
	      (let ((ring_buffer (static_cast<char*>
				  (mmap 0
					(* ring_info.tp_block_size
					   ring_info.tp_block_nr)
					(logior PROT_READ PROT_WRITE)
					(logior MAP_SHARED MAP_LOCKED)
					sockfd 0)))))

	      (let ((current_block 0))
		(comments "packet processing loop")
		(while true
		       (let ((pfd (pollfd (space (curly (= .fd sockfd)
							(= .events POLLIN)))))))
		       (let ((pollresult (poll &pfd 1 -1)))
			 (when (< 0 pollresult)
			   (dotimes (frame_idx (static_cast<int> ring_info.tp_frame_nr))
			     (let ((hdr (reinterpret_cast<tpacket3_hdr*>
					 (+ ring_buffer
					    (* current_block ring_info.tp_block_size)
					    (* frame_idx ring_info.tp_frame_size)))))
			       (when (& TP_STATUS_USER
					hdr->hv1.tp_rxhash)
				 (let ((placement_address (+ (reinterpret_cast<char*> hdr)
							     hdr->tp_next_offset))
				       (videoLine (space new (paren placement_address)
							 (VideoLine))))
				   ,(lprint :vars `(videoLine->width)))
				 (setf hdr->hv1.tp_rxhash 0)
				 (comments "delete of videoLine not required as it is placement new and memory is in ring buffer")))
			     )
			   (do0 (comments "move to next block")
				  (setf current_block (% (+ current_block 1)
							 ring_info.tp_block_nr)))))
		       (comments "prevent busy wait")
		       (std--this_thread--sleep_for (std--chrono--microseconds 1))
		       )))
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
