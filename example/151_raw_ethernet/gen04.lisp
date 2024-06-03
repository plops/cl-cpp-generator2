(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list :more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/151_raw_ethernet/source04/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (load "util.lisp")


  (let* ((class-name `PacketReceiver)
	 (members0 `((:name callback :type ; PacketReceivedCallback
			    "std::function<void(const uint8_t*, size_t)>"
			    :initform nullptr
			    #+nil (lambda (data size)
			      (declare (capture "")
				       (type "const uint8_t*" data)
				       (type size_t size)))
		      :param t)
		     (:name sockfd :type int :initform -1)
		     (:name mmap-base :type void* :initform nullptr)
		     (:name mmap-size :type size_t :initform 0)
		     (:name if-name :type "std::string" :param t :initform (string "lo"))
		     (:name frame-size :type uint32_t :initform 128 :param t)
		     (:name block-size :type uint32_t :initform 4096 :param t)
		     (:name block-nr :type uint32_t :initform 1 :param t)
		     
		     (:name frame-nr :type  uint32_t)
		     (:name rx-buffer-cnt :type uint32_t :doc "The number of frames in the RX ring buffer.")
		     
		     ))
	 (members (loop for e in members0
			collect
			(destructuring-bind (&key name type param doc initform) e
			  `(:name ,name
			    :type ,type
			    :param ,param
			    :doc ,doc
			    :initform ,initform
			    :member-name ,(intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))
			    :param-name ,(when param
					   (intern (string-upcase (cl-change-case:snake-case (format nil "~a" name))))))))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name class-name
     :headers `()
     :header-preamble `(do0 (comments "header")
			    (include<> cstdint
				       string
				       vector
				       array
				       functional))
     :implementation-preamble
     `(do0
       (comments "implementation")
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
	))
     :code `(do0
	     
	     (defclass ,class-name ()
	       "public:"
	       ;"using PacketReceivedCallback = std::function<void(const uint8_t*, const size_t)>"
	       ;; handle positional arguments, followed by optional arguments
	       (defmethod ,class-name (&key ,@(remove-if
					       #'null
					       (loop for e in members
						     collect
						     (destructuring-bind (&key name type param doc initform param-name member-name) e
						       (when param
							 `(,param-name ,(if initform initform 0)))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (&key name type param doc initform param-name member-name) e
				       (let ((const-p (let* ((s  (format nil "~a" type))
							     (last-char (aref s (- (length s)
										   1))))
							(not (eq #\* last-char)))))
					 (when param
					   (if (eq name 'callback)
					       `(type "std::function<void(const uint8_t*, const size_t)>"
						      #+nil PacketReceivedCallback ,param-name)
					       `(type ,(if const-p
						       (format nil "const ~a&" type)
						       type)
						  ,param-name))
					   )))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (&key name type param doc initform param-name member-name) e
					(cond
					  (param
					   (if (eq name 'callback)
					       `(,member-name (std--move ,param-name))
					       `(,member-name ,param-name))) 
					  (initform
					   `(,member-name ,initform)))))))
					;(explicit)	    
		  (values :constructor))
		 #+more ,(lprint :msg "PacketReceiver constuctor"
				 :vars `(if_name))
		 (do0
		  (setf sockfd (socket AF_PACKET
				  SOCK_RAW
				  (htons ETH_P_ALL)
				  ))
		  (when (< sockfd 0)
		    (throw (std--runtime_error
			    (string "error opening socket. try running as root"))))

		  (do0
		   (comments "bind socket to the hardware interface")
		   (let ((ifindex (static_cast<int> (if_nametoindex (if_name.c_str))))
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
		       (throw (std--runtime_error
			       (string "bind error"))))
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
		   (setf frame_nr (* (/ block_size frame_size)
				     block_nr))
		   (let ((req (space tpacket_req (curly (= .tp_block_size block_size)
							(= .tp_block_nr block_nr)
							(= .tp_frame_size  frame_size)
							(= .tp_frame_nr frame_nr)
							))))
		     (do0
		      (comments
		       "The following conditions are not strictly necessary for the kernel's ring buffer to function. However, the code that iterates through the blocks of the ring buffer is designed to handle these specific configurations.")
		      
		      (when (!= 0 (% block_size (getpagesize)))
			(throw (std--runtime_error (string "block_size should be a multiple of getpagesize()"))))
		      (comments "Calculate the factor as the ratio of block_size to page size.")

		      #+nil
		      (let ((factor (/ block_size
				       (getpagesize))))
			(comments 
			 "Check if this factor is a power of two. A number is a power of two if, in binary, it has exactly one bit set to 1. When you subtract 1 from such a number, all the bits after the 1 bit become 1. Therefore, a bitwise AND operation (&) between the number and (number - 1) will be zero.")
			(unless (!= 0 (logand factor (- factor 1)))
			  (throw (std--runtime_error (string "block_size/pagesize should be  a power of two")))))
		      (when (!= 0 (% block_size
				     frame_size))
			(throw (std--runtime_error (string "block_size should be a multiple of frame_size")))))
		     
		     ,(lprint :vars `(block_size block_nr frame_size frame_nr))
		     (setsockopt sockfd SOL_PACKET
				 PACKET_RX_RING
				 &req
				 (sizeof req))))

		  (do0
		   (comments "map the ring buffer")
		   (setf mmap_size (* block_size block_nr))
		   (setf mmap_base (mmap nullptr
					 mmap_size (or PROT_READ PROT_WRITE)
					 MAP_SHARED sockfd 0))
	      
		   (let ((rx_buffer_size (* block_size block_nr))
			 (rx_buffer_addr mmap_base)
			 )
		     (comments "rx_buffer_cnt is the number of frames in the ring buffer")
		     (setf rx_buffer_cnt (/ (* block_size block_nr)
					    frame_size))
		     ,(lprint :vars `(rx_buffer_size
				      rx_buffer_cnt)))))
		 )

	       (defmethod ,(format nil "~~~a" class-name) ()
		 (declare
		  (values :constructor))
		 (comments "Disable PACKET_RX_RING")
		 (let ((req (space tpacket_req (curly (= .tp_block_size 0)
						      (= .tp_block_nr 0)
						      (= .tp_frame_size  0)
						      (= .tp_frame_nr 0)
						      ))))
		   
		   (setsockopt sockfd SOL_PACKET
			       PACKET_RX_RING
			       &req
			       (sizeof req)))
		 (comments "Unmap the memory-mapped buffer")
		 (munmap mmap_base mmap_size)
		 (comments "Close the socket")
		 (close sockfd)
		 )
	       (doc
		"
@brief Continuously reads packets from a socket and processes them.

This function runs an infinite loop that continuously polls a socket for incoming packets.
When a packet is received, it is processed and the provided callback function is called with the packet data.
The function also handles various status flags and errors, and provides detailed logging of packet information and status.

The function uses a ring buffer mechanism for efficient packet reading. It keeps track of the current index in the ring buffer,
and after processing each packet, it hands the buffer back to the kernel and moves to the next buffer in the ring.

If the function encounters an error during polling, it throws a runtime_error exception.

If no packets are available for reading, the function sleeps for 4 milliseconds before polling the socket again.

@note This function runs indefinitely until an error occurs or the program is terminated.
 
")
	       (defmethod handlePackets ()
		 (do0
		  #+more ,(lprint :msg "handlePackets started"
				  :vars `(sockfd))
		  (let ((idx 0U)
			(old_arrival_time64 (uint64_t 0))))
		  (while true
			 (let ((pollfds (pollfd (curly (space (= .fd sockfd))
						       (space (= .events POLLIN))
						       (space (= .revents 0)))))
			       (poll_res (ppoll &pollfds 1 nullptr nullptr))) ;; waits indefinetely
			   (cond
			     ((< poll_res 0)
			      #+more ,(lprint :msg "error in ppoll"
					      :vars `(poll_res errno))
			      (throw (std--runtime_error
				      (string "ppoll error")))
			      )
			     ((== poll_res 0)
			      #+more ,(lprint :msg "timeout")
			      (std--this_thread--sleep_for (std--chrono--milliseconds 4)))
			     (t
	      		      (do0
			       (when (and POLLIN pollfds.revents)
				 (let ((base (+ (reinterpret_cast<uint8_t*> mmap_base)))
				       (header (reinterpret_cast<tpacket2_hdr*> (+ base
										   (* idx frame_size))))))
				 (comments "Iterate through packets in the ring buffer")
				 (space do
					(progn
				     	  (cond
					    ((and header->tp_status TP_STATUS_USER)
					     (do0
					      (cond
						#+more ((and header->tp_status TP_STATUS_COPY)
							,(lprint :msg "copy" :vars `(idx)))
						#+more ((and header->tp_status TP_STATUS_LOSING)
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
						    ))
					      #+more (let ((arrival_time64 (+ (* 1000000000 header->tp_sec)
									      header->tp_nsec))
							   (delta64 (- arrival_time64 old_arrival_time64 ))
							   (delta_ms (/ (static_cast<double> delta64) 1000000d0))
							   (arrival_timepoint (+ (std--chrono--system_clock--from_time_t header->tp_sec)
										 (std--chrono--nanoseconds  header->tp_nsec)))
							   (time (std--chrono--system_clock--to_time_t arrival_timepoint))
							   (local_time (std--localtime &time))
							   (local_time_hr (std--put_time local_time (string "%Y-%m-%d %H:%M:%S")))))
					
					      #+more (<< std--cout
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
							 )
					      #+more ,(let ((pos-red #+ipv6 `(,(- #x3b 13) 1)
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
					      #+more (do0
						      (<< std--cout std--dec std--endl)
						      (setf  old_arrival_time64  arrival_time64 ))
					      )
					     (callback data data_len)
					     (do0 (comments "Hand this entry of the ring buffer (frame) back to kernel")
						  (setf header->tp_status TP_STATUS_KERNEL))
					     )
					    (t
					     (comments "this packet is not tp_status_user, poll again")
					     #+more ,(lprint :msg "poll")
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
			   ))))
	       #-nil
	       ,@(remove-if
		  #'null
	          (loop for e in members
			appending
			(destructuring-bind (&key name type param doc initform param-name member-name) e
			  (let ((get (cl-change-case:pascal-case (format nil "get-~a" name)))
				(set (cl-change-case:pascal-case (format nil "set-~a" name)))
				(const-p (let* ((s  (format nil "~a" type))
						(last-char (aref s (- (length s)
								      1))))
					   (not (eq #\* last-char)))))
			    `(,(if doc
				   `(doc ,doc)
				   "")
			      (defmethod ,get ()
				(declare ,@(if const-p
					       `((const)
						 (values ,(format nil "const ~a&" type)))
					       `((values ,type))))
				(return ,member-name))
			      (defmethod ,set (,member-name)
				(declare (type ,type ,member-name))
				(setf (-> this ,member-name)
				      ,member-name)))))))
	       "private:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (&key name type param doc initform param-name member-name) e
				    `(space ,type ,member-name))))))))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<> iostream
		system_error
		unistd.h)
     (include "PacketReceiver.h")
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       ,(lprint :vars `(argc (aref argv 0)))
       (handler-case
	   (do0
	    (let ((cb (lambda (data size)
				       (declare (capture "")
						(type "const uint8_t*" data)
						(type size_t size))
			(when (< 0 size)
			 (<< std--cout (static_cast<int> (aref data 0))
			     std--endl))
				       )))
	      ;(declare (type "PacketReceiver::PacketReceivedCallback" cb))
	      )
	    (let ((name (std--string (string "lo")))
		  (block_size (getpagesize))
		  (block_nr 1)
		  (frame_size 128)
		  ))
	    ,(lprint :vars `(block_size))
	    (let ((r (PacketReceiver cb name frame_size block_size block_nr))))
	    (r.handlePackets)
	    
	    )
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
