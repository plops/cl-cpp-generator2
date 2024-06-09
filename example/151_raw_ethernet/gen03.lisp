(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    :ipv6
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						 ;:ipv6
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
     (doc "This C++ code is a low-level network packet sniffer that uses raw
sockets and the Linux packet_mmap API to capture packets from a
network interface.

The main function starts by creating a raw socket and binding it to
the `wlan0` interface. It then sets the socket option to use version 2
of the packet_mmap API (TPACKET_V2).

Next, it configures a ring buffer for packet capture, ensuring that
the block size is a multiple of the system page size and a power of
two, and that it is also a multiple of the frame size.

The ring buffer is then memory-mapped into the process's address
space.

The main loop of the program polls the socket for incoming
packets. When a packet arrives, it iterates through the packets in the
ring buffer, printing out information about each packet, including the
time of arrival, the length of the packet data, and a hex dump of the
first 128 bytes of the packet data.

If a packet is marked as having been copied or lost, it prints out
additional information.

After processing a packet, it hands the frame back to the kernel and
moves on to the next frame in the ring buffer.

If an error occurs during the execution of the program, it catches the
exception and prints out an error message.")
     
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
      
					;cstring
      iostream
					;array
					;span
      system_error
					;format
      cstdint
					;cstring
					;chrono
      thread
      iomanip
      )
     #+nil
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

	    (when (< sockfd 0)
	      ,(lprint :msg "error opening socket. try running as root")
	      (return -1))

	    
	    (do0
	     (comments "bind socket to the hardware interface")
	     (let ((ifindex (static_cast<int> (if_nametoindex (string #+nil "lo"
								      #-nil "wlan0"))))
		   (ll (sockaddr_ll (curly (= .sll_family AF_PACKET)
					   (= .sll_protocol (htons ETH_P_ALL))
					   (= .sll_ifindex ifindex)
					   (= .sll_hatype ARPHRD_ETHER)
					   (= .sll_pkttype (or PACKET_HOST
							       PACKET_OTHERHOST
							       PACKET_BROADCAST))
					   (= .sll_halen 0)
					   (= .sll_addr (curly 0 0 0 0  0 0 0 0))
					   ))))
	       ,(lprint :vars `(ifindex))
	       (when (< (bind sockfd (reinterpret_cast<sockaddr*> &ll)
			  (sizeof ll))
			0)
		 ,(lprint :msg "bind error" :vars `(errno)))
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
		   (block_nr 8U)
		   (frame_size 256U
			       ;2048U
			       )
		   (frame_nr (* (/ block_size frame_size)
				block_nr))
		   (req (space tpacket_req (curly (= .tp_block_size block_size)
						  (= .tp_block_nr block_nr)
						  (= .tp_frame_size  frame_size)
						  (= .tp_frame_nr frame_nr)
						  ))))
	       (do0
		(comments "the following conditions don't have to be strictly fulfilled. the ring buffer"
			  "in the kernel works with other configurations. but my code to iterate through"
			  "the blocks of the ring buffer can only handle this")
		
		(when (!= 0 (% block_size (getpagesize)))
		  (throw (std--runtime_error (string "block_size should be a multiple of getpagesize()"))))
		#+nil (let ((factor (/ block_size
				 (getpagesize))))
		  (comments "if factor is a power of two, it has exactly one bit set to 1 in its binary representation"
			    "when you subtract 1 from factor, all the bits after the bit that was set to 1 are set to 1"
			    "the bitwise AND operation returns a number that has 1s in the positions where both numbers have 1s"
			    "if factor is a power of two, then factor & (factor-1) is zero")
		  (unless (!= 0 (logand factor (- factor 1)))
		    (throw (std--runtime_error (string "block_size/pagesize should be  a power of two")))))
		(when (!= 0 (% block_size
			       frame_size))
		  (throw (std--runtime_error (string "block_size should be a multiple of frame_size")))))
	       

	       ,(lprint :vars `(block_size block_nr frame_size frame_nr))
	       (when (< 
		      (setsockopt sockfd SOL_PACKET
				  PACKET_RX_RING
				  &req
				  (sizeof req))
		      0)
		 (throw (std--runtime_error (string "setsockopt"))))))

	    (do0
	     (comments "map the ring buffer")
	     (let ((mmap_size (* block_size block_nr))
		   (mmap_base (mmap nullptr
				    mmap_size (or PROT_READ PROT_WRITE)
				    MAP_SHARED sockfd 0))))
	     (let ((rx_buffer_size (* block_size block_nr))
		   (rx_buffer_addr mmap_base)
		   (rx_buffer_cnt (/ (* block_size block_nr)
				     frame_size)))
	       ,(lprint :vars `(rx_buffer_size
				rx_buffer_cnt))))

	    (let ((idx 0U)
		  (old_arrival_time64 (uint64_t 0))))
	    (while true
		   (let ((pollfds (pollfd (curly (space (= .fd sockfd))
						 (space (= .events POLLIN))
						 (space (= .revents 0)))))
			 (poll_res (ppoll &pollfds 1 nullptr nullptr))) ;; waits indefinetely
		     (cond
		       ((< poll_res 0)
			,(lprint :msg "error in ppoll"
				 :vars `(poll_res errno))
			)
		       ((== poll_res 0)
			,(lprint :msg "timeout")
			(std--this_thread--sleep_for (std--chrono--milliseconds 4)))
		       (t
	      		(do0
			 
			 (when (and POLLIN pollfds.revents)
			 
			   
			   (let ((base (+ (reinterpret_cast<uint8_t*> rx_buffer_addr)))
				 (header (reinterpret_cast<tpacket2_hdr*> (+ base
									     (* idx frame_size))))))
			   (comments "Iterate through packets in the ring buffer")
			   (space do
				  (progn
				    
				    (cond
				      ((and header->tp_status TP_STATUS_USER)
				       (do0
					
					(cond
					  ((and header->tp_status TP_STATUS_COPY)
					   ,(lprint :msg "copy" :vars `(idx)))
					  ((and header->tp_status TP_STATUS_LOSING)
					   (let ((stats (tpacket_stats))
						 (stats_size (static_cast<socklen_t> (sizeof stats))))
					     (getsockopt sockfd SOL_PACKET
							 PACKET_STATISTICS
							 &stats
							 &stats_size))
					   
					   ,(lprint :msg "loss" :vars `(idx stats.tp_drops
									    stats.tp_packets)))
					  )
					(let ((data (+ (reinterpret_cast<uint8_t*> header) header->tp_net))
					      (data_len header->tp_snaplen)
					      (arrival_time64 (+ (* (uint64_t 1000000000) header->tp_sec)
							       header->tp_nsec))
					      (delta64 (- arrival_time64 old_arrival_time64 ))
					      ))
					(<< std--cout
					    std--dec
					    (std--setw 20)
					    (std--setfill (char " "))
					    arrival_time64
					    (string " ")
					    (std--setw 12)
					    delta64
					    (string " "))
					;,(lprint :vars `(arrival_time64 delta64))
					#+nil (let ((delta_ms (/ (static_cast<double> delta64) 1000000d0))
						    (arrival_timepoint (+ (std--chrono--system_clock--from_time_t header->tp_sec)
									  (std--chrono--nanoseconds  header->tp_nsec)))
						    (time (std--chrono--system_clock--to_time_t arrival_timepoint))
						    (local_time (std--localtime &time))
						    (local_time_hr (std--put_time local_time (string "%Y-%m-%d %H:%M:%S"))))
					  
						(<< std--cout
						    local_time_hr
						    (string ".")
						    std--dec
						    (std--setw 6) ;; i want 1us granularity
						    (/ header->tp_nsec 1000)
						    (string " ")
						    (std--setfill (char " "))
						    (std--setw (+ 5 6))
						    std--fixed
						    (std--setprecision 6)
						    (? (< delta_ms 10000) (std--to_string delta_ms) (string "xxxx.xxxxxx"))
						    (string " ")
						    std--dec
						    (std--setw 6)
						    data_len
						    (string " ")
						    (std--setw 4)
						    idx
						    (string " ")
						    ))
					,(let ((pos-red #+ipv6 `(,(- #x3b 13) 1)
							#-ipv6 `(,(- 40 13) 1)))
					   `(dotimes (i (? (< data_len 128U)
							   data_len
							   128U))
					      (declare (type "unsigned int" i))
					      (comments "color sequence bytes of icmp packet in red")
					      (when (== ,(elt pos-red 0) i)
						(<< std--cout (string "\\033[31m"))
						)
					      (<< std--cout
						  std--hex
						  (std--setw 2)
						  (std--setfill (char "0"))
						  (static_cast<int> (aref data i)))
					      (when (== (+ ,(elt pos-red 0)
							   ,(elt pos-red 1) )
							i)
						(<< std--cout (string "\\033[0m")))
					      (when (== 7 (% i 8))
						(<< std--cout (string " "))))

					   )
					(do0
					    (<< std--cout std--dec std--endl)
					    (setf  old_arrival_time64  arrival_time64 ))
					)

				       (do0 (comments "Hand this entry of the ring buffer (frame) back to kernel")
					    (setf header->tp_status TP_STATUS_KERNEL))
				       )
				      (t
				       (comments "this packet is not tp_status_user, poll again")
				       ,(lprint :msg "poll")
				       continue))
				    (do0 
				     (comments "Go to next frame in ring buffer")
				     (setf idx (% (+ idx 1)
						  rx_buffer_cnt))
				     (setf  header (reinterpret_cast<tpacket2_hdr*>
						    (+ base
						       (* idx frame_size))))))
				  while
				  (paren (and header->tp_status TP_STATUS_USER)) ))
			 (std--this_thread--sleep_for (std--chrono--milliseconds 4))
			 )))
		     )))
	 ("const std::system_error&" (ex)
	   (<< std--cerr
	       (string "Error: ")
	       (ex.what)
	       (string " (")
	       (ex.code)
	       (string ")\\n"))
	   (return 1)))

       (comments "unreachable:")
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
