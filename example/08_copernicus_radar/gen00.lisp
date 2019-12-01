(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

;; https://docs.google.com/presentation/d/1LAm3p20egBVvj86p_gmaf-zuPYm4_OAKut5xcl9cWwk/edit?usp=sharing


;;  S1A_EW_GRDM_1SDH_20191130T152919_20191130T153018_030142_0371AB_6678
;;  S1A_EW_RAW__0SDH_20191130T152915_20191130T153018_030142_0371AB_8C38
(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/08_copernicus_radar/source/")
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames #P"run_01_base.c"
											       *source-dir*)))
  
  (progn
    (defun next-power-of-two (n)
      (let ((i 1))
	(loop while (< i n) do
	     (setf i (* i 2)))
	i))

    (defparameter *space-packet* ;; byte-order big-endion
      `(
	;; start of packet primary header
	(packet-version-number 0 :bits 3)
	(packet-type 0 :bits 1)
	(secondary-header-flag 0 :bits 1)
	(application-process-id-process-id 0 :bits 7)
	(application-process-id-packet-category 0 :bits 4) 
	(sequence-flags 0 :bits 2)
	(sequence-count 0 :bits 14) ;; 0 at start of measurement, wraps after 16383
	(data-length 0 :bits 16) ;; (number of octets in packet data field)
	;; - 1, includes 62 octets of secondary
	;; header and the user data field of
	;; variable length start of packet data
	;; field datation service p. 15
	(coarse-time 0 :bits 32)
	(fine-time 0 :bits 16)
	;; fixed ancillary data service
	(sync-marker #x352EF853
		     :bits 32)
	(data-take-id 0 :bits 32)
	(ecc-number 0 :bits 8)
	(ignore-0 0 :bits 1)
	(test-mode 0 :bits 3)
	(rx-channel-id 0 :bits 4)
	(instrument-configuration-id 0 :bits 32)
	;; sub-commutation ancillary data service
	(sub-commutated-index 0 :bits 8) ;; 1..64 slowly fills datastructure
	;; on p23,24,25 0 indicates invalid data word consistent set only
	;; after contiguous sequence 1..22 (pvt), 23..41 (attitude) or
	;; 42..64 (temperatures)
	(sub-commutated-data 0 :bits 16) 
	;; counters service
	(space-packet-count 0 :bits 32) ;; from beginning of data take
	(pri-count 0 :bits 32)
	;; radar configuration support service
	(error-flag 0 :bits 1)
	(ignore-1 0 :bits 2)
	(baq-mode 0 :bits 5)
	(baq-block-length 0 :bits 8)
	(ignore-2 0 :bits 8)
	(range-decimation 0 :bits 8)
	(rx-gain 0 :bits 8)
	;;(tx-ramp-rate 0 :bits 16)
	(tx-ramp-rate-polarity 0 :bits 1)
	(tx-ramp-rate-magnitude 0 :bits 15)
	(tx-pulse-start-frequency-polarity 0 :bits 1)
	(tx-pulse-start-frequency-magnitude 0 :bits 15)
	(tx-pulse-length 0 :bits 24)
	(ignore-3 0 :bits 3)
	(rank 0 :bits 5)
	(pulse-repetition-interval 0 :bits 24)
	(sampling-window-start-time 0 :bits 24)
	(sampling-window-length 0 :bits 24)
	;;(sab-ssb-message 0 :bits 24)
	(sab-ssb-calibration-p 0 :bits 1)
	(sab-ssb-polarisation 0 :bits 3)
	(sab-ssb-temp-comp 0 :bits 2)
	(sab-ssb-ignore-0 0 :bits 2)
	(sab-ssb-elevation-beam-address 0 :bits 4) ;; if calibration-p=1 sastest caltype
	(sab-ssb-ignore-1 0 :bits 2)
	(sab-ssb-azimuth-beam-address 0 :bits 10)
	;;(ses-ssb-message 0 :bits 24)
	(ses-ssb-cal-mode 0 :bits 2)
	(ses-ssb-ignore-0 0 :bits 1)
	(ses-ssb-tx-pulse-number 0 :bits 5)
	(ses-ssb-signal-type 0 :bits 4) ;; 0 echo, 1 noise, 8 txcal, 9 rxcal
	(ses-ssb-ignore-1 0 :bits 3)
	(ses-ssb-swap 0 :bits 1)
	(ses-ssb-swath-number 0 :bits 8)
	;; radar sample count service
	(number-of-quads 0 :bits 16)
	(ignore-4 0 :bits 8)
	))

    ;;(defparameter slot-idx (position 'data-length *space-packet* :key #'first))
    (defun space-packet-slot-get (slot-name data8)
      (let* ((slot-idx (position slot-name *space-packet* :key #'first))
	     (preceding-slots (subseq *space-packet* 0 slot-idx))
	     (sum-preceding-bits (reduce #'+
					 (mapcar #'(lambda (x)
						     (destructuring-bind (name_ default-value &key bits) x
						       bits))
						 preceding-slots)))
	     )
	(multiple-value-bind (preceding-octets preceding-bits) (floor sum-preceding-bits 8)
	  (destructuring-bind (name_ default-value &key bits) (elt *space-packet* slot-idx)
	    (assert (<= 0 preceding-bits 7))
	    (assert (<= 0 preceding-octets))
	    ;(format t "~a ~a ~a ~a~%" preceding-octets preceding-bits bits default-value)
	    (if (<= (+ preceding-bits bits) 8)
		(let ((mask 0)
		      (following-bits (- 8 (+ preceding-bits bits))))
		  (declare (type (unsigned-byte 8) mask))
		  (assert (<= 0 following-bits 8))

		  (setf (ldb (byte bits 0 ;(- 8 (+ bits preceding-bits))
				   ) mask) #xff)
		  (values
		   
		   `(&
		   #+nil  (string ,(format nil "single mask=~x following-bits=~d bits=~d preceding-bits=~d"
				      mask following-bits bits preceding-bits))
		     (hex ,mask)
		     (>> (aref ,data8 ,preceding-octets)
			 ,following-bits))
		   'uint8_t
		   ))
		
		(multiple-value-bind (bytes rest-bits) (floor (+ preceding-bits bits) 8)
		  (let* ((firstmask 0)
			 (lastmask 0)
			 (following-bits (- 8 rest-bits))
			 (first-bits (- 8 preceding-bits))
			 (last-bits (mod (- bits first-bits) 8)))
		    (assert (<= 1 first-bits 8))
		    (assert (<= 0 last-bits 8))
		    #+nil
		    (break "following-bits=~d rest-bits=~d bits=~d preceding-bits=~d bytes=~d first-bits=~d last-bits=~d"
			   following-bits rest-bits bits preceding-bits bytes first-bits last-bits)
		    (setf (ldb (byte first-bits 0) firstmask) #xff
			  (ldb (byte last-bits (- 8 last-bits)) lastmask) #xff)
		    (values
		     (if (= lastmask 0)
			 `(+
			   #+nil  (string ,(format nil "nolast firstmask=~x following-bits=~d rest-bits=~d bits=~d preceding-bits=~d bytes=~d first-bits=~d last-bits=~d"
						   firstmask  following-bits rest-bits bits preceding-bits bytes
						   first-bits last-bits)
					  )
			   ,@(loop for byte from (- bytes 1) downto 1 collect
				  `(* (hex ,(expt 256 (- bytes byte 1)))
				      (aref ,data8 ,(+ preceding-octets 0 byte))))
			   (* (hex ,(expt 256 (- bytes 1))) (& (hex ,firstmask) (aref ,data8 ,(+ preceding-octets 0))
							       )))
			 `(+
			   #+nil  (string ,(format nil "both firstmask=~x lastmask=~x following-bits=~d rest-bits=~d bits=~d preceding-bits=~d bytes=~d first-bits=~d last-bits=~d"
						   firstmask lastmask following-bits rest-bits bits preceding-bits bytes first-bits last-bits))
			   (>> (& (hex ,lastmask) (aref ,data8 ,(+ preceding-octets 0 bytes)))
			       ,following-bits)
			   ,@(loop for byte from (- bytes 1) downto 1 collect
				  `(* (hex ,(expt 256 (- bytes byte)))
				      (aref ,data8 ,(+ preceding-octets 0 byte))))
			   (* (hex ,(expt 2 (1- preceding-bits))	;,(expt 256 bytes)
			       ) (& (hex ,firstmask) (aref ,data8 ,(+ preceding-octets 0))))))
		     (format nil "uint~a_t" (next-power-of-two bits))))
		  ))))))


    )
  
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun vkprint (call &optional rest)
      `(do0 ,call
            (<< "std::cout"
              (- (dot ("std::chrono::high_resolution_clock::now")
                    (time_since_epoch)
                    (count))
                 ,(g `_start_time))
              (string " ")
              __FILE__
              (string ":")
              __LINE__
              (string " ")
              __func__
              (string ,(format nil " ~a " (emit-c :code call)))
              ,@(loop for e in rest appending
                     `((string ,(format nil " ~a=" (emit-c :code e)))
                       ,e))
              "std::endl")))
    (defun logprint (msg &optional rest)
      `(do0
	("std::setprecision" 3)
	(<< "std::cout"
	    ("std::setw" 10)
	       (- (dot ("std::chrono::high_resolution_clock::now")
		       (time_since_epoch)
		       (count))
		  ,(g `_start_time))
	       (string " ")
	       __FILE__
	       (string ":")
	       __LINE__
	       (string " ")
	       __func__
	       (string " ")
	       (string ,msg)
	       (string " ")
	       ,@(loop for e in rest appending
		      `(("std::setw" 8)
					;("std::width" 8)
			(string ,(format nil " ~a=" (emit-c :code e)))
			,e))
	       "std::endl")))

    (defun gen-huffman-decoder (name huffman-tree)
      (labels ((frob (tree)
	     (cond ((null tree)
		    (error "null"))
		   ((atom tree) `(return ,tree))
		   ((null (cdr tree))
		    `(return ,(car tree)))
		   (t `(if (get_sequential_bit s)
			   ,(frob (cadr tree))
			   ,(frob (car tree))
			   )))))
       `(defun ,(format nil "decode_huffman_~a" name) (s)
	  (declare (type sequential_bit_t* s)
		   (values "inline int"))
	  ,(frob huffman-tree)
	  )))

    (defun emit-globals (&key init)
      (let ((l `(
		 (_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
                                                                                       (time_since_epoch)
                                                                                       (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  `(,name ,type)))
		 )))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
			 (destructuring-bind (name type &optional value) e
			   `(,name ,type))))))))
  
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	
	  (push `(do0
		
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
			
		  " "
		  )
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header)
	    )
	
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))
  (define-module
      `(main ((_filename :direction 'out :type "char const *"))
	     (do0
	      (include <iostream>
		       <chrono>
		       )
	      (let ((state ,(emit-globals :init t)))
		(declare (type State state)))
	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))
					;(vkprint "main" )
		(setf ,(g `_filename)
		      (string "/home/martin/Downloads/s1a-ew-raw-s-hv-20191130t152915-20191130t153018-030142-0371ab.dat"
			      ;"/home/martin/Downloads/S1A_IW_RAW__0SDV_20181106T135244_20181106T135316_024468_02AEB9_3552.SAFE/s1a-iw-raw-s-vh-20181106t135244-20181106t135316-024468-02aeb9.dat"
			      ;"/home/martin/Downloads/S1A_IW_RAW__0SDV_20191125T135230_20191125T135303_030068_036F1E_6704.SAFE/s1a-iw-raw-s-vv-20191125t135230-20191125t135303-030068-036f1e.dat"
			      )) 
		(init_mmap ,(g `_filename))
		(init_collect_packet_headers) 
					;(init_process_packet_headers)
		(let ((output)
		      (n (init_decode_packet 0 output)))
		  (declare (type "std::array<std::complex<float>,65535>" output)))
		;(init_decode_packet 1)
		;(init_decode_packet 2)
		(destroy_mmap)
		))))
  (define-module
      `(mmap
	((_mmap_data :direction 'out :type void*)
	 (_mmap_filesize :direction 'out :type size_t))
	(do0
	 (include <sys/mman.h>
		  <sys/stat.h>
		  <sys/types.h>
		  <fcntl.h>
		  <cstdio>
		  <cassert>
		  <iostream>)
	 (defun get_filesize (filename)
	   (declare (type "const char*" filename)
		    (values size_t))
	   (let ((st))
	     (declare (type "struct stat" st))
	     (stat filename &st)
	     (return st.st_size)))
	 (defun destroy_mmap ()
	   (let ((rc (munmap ,(g `_mmap_data)
			     ,(g `_mmap_filesize))))  
	     (unless (== 0 rc)
	       ,(logprint "fail munmap" `(rc)))
	     (assert (== 0 rc))))
	 (defun init_mmap (filename)
	   (declare (type "const char*" filename)
		    )
	   (let ((filesize (get_filesize filename))
		 (fd (open filename O_RDONLY 0)))
	     ,(logprint "size" `(filesize filename))
	     (when (== -1 fd)
	       ,(logprint "fail open" `(fd filename)))
	     (assert (!= -1 fd))
	     (let ((data (mmap NULL filesize PROT_READ MAP_PRIVATE fd 0)))
	       (when (== MAP_FAILED data)
		 ,(logprint "fail mmap"`( data)))
	       (assert (!= MAP_FAILED data))
	       
	       (setf ,(g `_mmap_filesize) filesize
		     ,(g `_mmap_data) data)))
	   ))))

  (define-module
      `(collect_packet_headers
	(;(_header_data :direction 'out :type void*)
	 (_header_data :direction 'out :type "std::vector<std::array<uint8_t,62+6>>")
	 (_header_offset :direction 'out :type "std::vector<size_t>")
	 )
	(do0
	 (include <array>
		  <iostream>
		  <vector>
		  <cstring>
		  )
	 #+nil (defstruct0 space_packet_header_info_t
	   (head "std::array<uint8_t, 62+6>")
	   (offset size_t))
	 (defun destroy_collect_packet_headers ()
	 )
	 (defun init_collect_packet_headers ()
	   ,(logprint "collect" `(,(g `_mmap_data)))
	   (let ((offset 0))
	     (declare (type size_t offset))
	     (while (< offset ,(g `_mmap_filesize)) ;dotimes (i 2000)
	       (let ((p (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
		     (data_length ,(space-packet-slot-get 'data-length `p)
		       )
		     ;(sync_marker ,(space-packet-slot-get 'sync-marker `p))
		     (data_chunk)
		     )
		 (declare (type "std::array<uint8_t,62+6>" data_chunk))
		 (memcpy (data_chunk.data)
			 p
			 (+ 62 6))
		 #+nil ,(logprint "len" `(offset data_length ;sync_marker
					 ))
		 
		 (dot ,(g `_header_offset)
		      (push_back offset))
		 (dot ,(g `_header_data)
		      (push_back data_chunk))
		 (incf offset (+ 6 1 data_length)))))
	   ))))

  (define-module
      `(process_packet_headers
	()
	(do0
	 (include <unistd.h>)
	 
	 (defun init_process_packet_headers ()
	   (let ((p0 (dot (aref ,(g `_header_data) 0)
			  (data)))
		 (coarse_time0 ,(space-packet-slot-get 'coarse-time 'p0))
		 (fine_time0 (* ,(expt 2d0 -16) (+ .5 ,(space-packet-slot-get 'fine-time 'p0))))
		 (time0 (+ coarse_time0 fine_time0))
		 (packet_idx 0))
	     (foreach (e ,(g `_header_data))
		      (let ((offset (aref ,(g `_header_offset) packet_idx))
			      (p (+ offset (static_cast<uint8_t*> ,(g `_mmap_data)))))
			  (incf packet_idx))
		     (let (;(p (e.data))
			   (fref 37.53472224)
			   (coarse_time ,(space-packet-slot-get 'coarse-time 'p))
			   (fine_time ,(space-packet-slot-get 'fine-time 'p))
			   (ftime (* ,(expt 2d0 -16) (+ .5 fine_time)))
			   (time (- (+ coarse_time
				       ftime)
				    time0))
			   (swst (/ ,(space-packet-slot-get 'sampling-window-start-time 'p)
				    fref))
			   (azi ,(space-packet-slot-get 'sab-ssb-azimuth-beam-address 'p))
			   (count ,(space-packet-slot-get 'space-packet-count 'p))
			   (pri_count ,(space-packet-slot-get 'pri-count 'p))
			   (pri (/ ,(space-packet-slot-get 'pulse-repetition-interval 'p)
				   fref))
			   (rank ,(space-packet-slot-get 'rank 'p))
			   (rank2 (static_cast<int> (aref p (+ 49))))
			   (baqmod ,(space-packet-slot-get 'baq-mode 'p))
			    (baq_n ,(space-packet-slot-get 'baq-block-length 'p))
			   (sync_marker ,(space-packet-slot-get 'sync-marker 'p))
			   (sync2 (+ ,@(loop for j below 4  collect
					    `(* ,(expt 256 (- 3 j)) (logand #xff (static_cast<int> (aref p (+ 12 ,j))))))))
			   (baqmod2 (static_cast<int> (aref p 37))  ;(logand #x1F (>> (aref p 37) 3))
			     )
			   (err ,(space-packet-slot-get 'error-flag 'p))
			   (tstmod ,(space-packet-slot-get 'test-mode 'p))
			   (rx ,(space-packet-slot-get 'rx-channel-id 'p))
			   (ecc ,(space-packet-slot-get 'ecc-number 'p))
			   (pol ,(space-packet-slot-get 'sab-ssb-polarisation 'p))
			   (signal_type ,(space-packet-slot-get 'ses-ssb-signal-type 'p))
			   (swath ,(space-packet-slot-get 'ses-ssb-swath-number 'p))
			   (ele ,(space-packet-slot-get 'sab-ssb-elevation-beam-address 'p)))

		       ,@(loop for e in *space-packet* collect
			      (destructuring-bind (name_ default-value &key bits) e
				
				`(progn
				   (let ((v (static_cast<int> ,(space-packet-slot-get name_ 'p))))
				     (<<
				      "std::cout"
				      ("std::setw" 42 )
				      (string ,(format nil "~a " name_))
				      ("std::setw" 12)
				      "std::dec"
					    v
					    
					    ("std::setw" 12)
					    "std::hex"
					    v
					    (string " ")
					    
					    ,@ (let ((bits (destructuring-bind (name default &key bits) (find name_  *space-packet*    :key #'first)
							     bits)))
						 (loop for j from (1- bits) downto 0 collect
						      `(static_cast<int> (logand 1 (>> v ,j)))))
					       
					    "std::endl"
					    )))))

		       (do0 (dotimes (i (+ 6 62))
			      "// https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal"
			      "// dump binary"
			      (<< "std::cout" (string "\\033[")
				  "std::dec"
				  (+ 30 (% (- (+ 7 6 62) i ) (- 37 30)))
				  (string ";")
				  (+ 40 (% i (- 47 40)))
				  (string "m")
				  ;"std::hex" ("std::setw" 2)
					;(static_cast<int> (aref p i))
				  ,@(loop for j from 7 downto 0 collect
					 `(static_cast<int> (logand 1 (>> (aref p i) ,j))))
				  (string "\\033[0m")
				  (string " "))
			      (when (== 3 (% i 4))
				(<< "std::cout" "std::endl")))
			    (<< "std::cout" (string "\\033[0m") "std::endl" "std::flush"))
		      
		       #+nil ,(logprint "" `(time "std::hex" err
					    swst coarse_time fine_time swath count pri_count rank rank2 pri baqmod baq_n sync2 sync_marker baqmod2 tstmod azi ele
					    rx pol ecc signal_type
					    ))
		       (usleep 16000)
		       (<< "std::cout"
			   (string "\\033[2J\\033[1;1H")
			   "std::flush")
		       )))
	   ))))

  (define-module
      `(decode_packet
	()
	(do0
	 (do0
	  (include <cassert>
		   <cmath>)
	  ,(emit-utils :code
		       `(do0
			 (include <complex>)
			 (defstruct0 sequential_bit_t
					;(user_data_position size_t)
			     (current_bit_count size_t)
					;(current_byte size_t)
			   (data uint8_t*)
			   )))
	  #+nil (defun reverse_bit (b)
	    (declare (type uint8_t b)
		     (values uint8_t))
	    
	    "// http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64BitsDiv"
	    "// b = ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;"
	    (return (>> (* (logand (* b "0x80200802ULL")
				   "0x0884422110ULL")
			   "0x0101010101ULL")
			32)))
	  (defun init_sequential_bit_function (seq_state byte_pos)
	    (declare (type sequential_bit_t* seq_state)
		     (type size_t byte_pos))
	    (setf seq_state->data (ref (aref (static_cast<uint8_t*> ,(g `_mmap_data))
					     byte_pos))
		  seq_state->current_bit_count 0))
	  (defun get_sequential_bit (seq_state)
	    (declare (type sequential_bit_t* seq_state)
		     (values "inline bool"))
	    (let ((current_byte (deref seq_state->data))
		  (res (static_cast<bool>
			(logand (>> current_byte (- 7 seq_state->current_bit_count))
				1))))
	      (incf seq_state->current_bit_count)
	      (when (< 7 seq_state->current_bit_count)
		(setf seq_state->current_bit_count 0)
		(incf seq_state->data))
	      (return res))))
	 (defun get_threshold_index (s)
	   (declare (type sequential_bit_t* s)
		    (values "inline int"))
	   (return (+ ,@(loop for j below 8 collect
			     `(* (hex ,(expt 2 (- 7 j)))
				 (get_sequential_bit s)))))
	   )
	 (defun get_bit_rate_code (s)
	   (declare (type sequential_bit_t* s)
		    (values "inline int"))
	   "// note: evaluation order is crucial"
	   #+nil(let ((a (get_sequential_bit s))
		 (b (get_sequential_bit s))
		 (c (get_sequential_bit s)))
	     
		  
		  )
	   #+nil (<<
	    "std::cout"
	    ,@ (let ((bits (destructuring-bind (name default &key bits) (find name_  *space-packet*    :key #'first)
			     bits)))
		 (loop for j from (1- bits) downto 0 collect
		      `(static_cast<int> (logand 1 (>> v ,j)))))
	       "std::endl")
	   
	   (return (+ ,@(loop for j below 3 collect
			     `(* (hex ,(expt 2 (- 2 j)))
				 (get_sequential_bit s))))))
	 (defun consume_padding_bits (s)
	   (declare (type sequential_bit_t* s)
		    (values "inline void"))
	   ;; fixme: mod on current_bit_count is not working
	   ;; check for odd/even byte instead
	   (let ((byte_offset (static_cast<int> (- s->data
						   (static_cast<uint8_t*> ,(g `_mmap_data))))))
	     "// make sure we are at first bit of an even byte in the next read"
	     (setf s->current_bit_count 0)
	     (if (== 0 (% byte_offset 2))
		 (do0
		  "// we are in an even byte"
		  (incf s->data 2))
		 (do0
		  "// we are in an odd byte"
		  (incf s->data 1))))
	   #+nil
	   (let ((pad (- 16
			 (% s->current_bit_count 16))))
	     ,(logprint "" `(pad))
	     (dotimes (i pad)
	       (assert (== 0
			(get_sequential_bit s))))))

	 ;; decode_huffman_brc<n> n=0..4
	 ,(gen-huffman-decoder 'brc0 '(0 (1 (2 (3))))) ;; page 71 in space packet protocol data unit
	 ,(gen-huffman-decoder 'brc1 '(0 (1 (2 (3 (4))))))
	 ,(gen-huffman-decoder 'brc2 '(0 (1 (2 (3 (4 (5 (6))))))))
	 ,(gen-huffman-decoder 'brc3 '((0 1) (2 (3 (4 (5 (6 (7 (8 (9))))))))))
	 ,(gen-huffman-decoder 'brc4 '((0 (1 2)) ((3 4) ((5 6) (7 (8 (9 ((10 11) ((12 13) (14 15))))))))))

	 #+nil (defun decode_symbol (s)
	   (declare (type sequential_bit_t* s)
		    (values "inline float"))
	   (return 0s0))
	 
	 
	 (do0
	  "// table 5.2-1 simple reconstruction parameter values B"
	  ,@(loop for l in `((3 3 3.16 3.53)
			     (4 4 4.08 4.37)
			     (6 6 6 6.15 6.5 6.88)
			     (9 9 9 9 9.36 9.5 10.1)
			     (15 15 15 15 15 15 15.22 15.5 16.05))
	       and brc from 0 collect
		 (let ((table (format nil "table_b~a" brc)))
		  `(let ((,table (curly ,@l)))
		     (declare (type ,(format nil "const std::array<float,~a>" (length l)) ,table))))))

	 (do0
	  "// table 5.2-2 normalized reconstruction levels"
	  ,@(loop for l in `((.3637 1.0915 1.8208 2.6406)
			     (.3042  .9127 1.5216 2.1313 2.8426)
			     (.2305  .6916 1.1528 1.6140 2.0754 2.5369 3.1191)
			     (.1702  .5107  .8511 1.1916 1.5321 1.8726 2.2131 2.5536 2.8942 3.3744)
			     (.1130  .3389  .5649  .7908 1.0167 1.2428 1.4687 1.6947 1.9206 2.1466 2.3725 2.5985
				     2.8244 3.0504 3.2764 3.6623))
	       and brc from 0 collect
		 (let ((table (format nil "table_nrl~a" brc)))
		  `(let ((,table (curly ,@l)))
		     (declare (type ,(format nil "const std::array<float,~a>" (length l)) ,table))))))
	 
	 
	 (do0
	  "// table 5.2-3 sigma factors"

	  ,(let ((l '(0.00 0.63 1.25 1.88 2.51 3.13 3.76 4.39 5.01 5.64 6.27
		     6.89 7.52 8.15 8.77 9.40 10.03 10.65 11.28 11.91 12.53 13.16 13.79
		     14.41 15.04 15.67 16.29 16.92 17.55 18.17 18.80 19.43 20.05 20.68
		     21.31 21.93 22.56 23.19 23.81 24.44 25.07 25.69 26.32 26.95 27.57
		     28.20 28.83 29.45 30.08 30.71 31.33 31.96 32.59 33.21 33.84 34.47
		     35.09 35.72 36.35 36.97 37.60 38.23 38.85 39.48 40.11 40.73 41.36
		     41.99 42.61 43.24 43.87 44.49 45.12 45.75 46.37 47.00 47.63 48.25
		     48.88 49.51 50.13 50.76 51.39 52.01 52.64 53.27 53.89 54.52 55.15
		     55.77 56.40 57.03 57.65 58.28 58.91 59.53 60.16 60.79 61.41 62.04
		     62.98 64.24 65.49 66.74 68.00 69.25 70.50 71.76 73.01 74.26 75.52
		     76.77 78.02 79.28 80.53 81.78 83.04 84.29 85.54 86.80 88.05 89.30
		     90.56 91.81 93.06 94.32 95.57 96.82 98.08 99.33 100.58 101.84 103.09
		     104.34 105.60 106.85 108.10 109.35 110.61 111.86 113.11 114.37
		     115.62 116.87 118.13 119.38 120.63 121.89 123.14 124.39 125.65
		     126.90 128.15 129.41 130.66 131.91 133.17 134.42 135.67 136.93
		     138.18 139.43 140.69 141.94 143.19 144.45 145.70 146.95 148.21
		     149.46 150.71 151.97 153.22 154.47 155.73 156.98 158.23 159.49
		     160.74 161.99 163.25 164.50 165.75 167.01 168.26 169.51 170.77
		     172.02 173.27 174.53 175.78 177.03 178.29 179.54 180.79 182.05
		     183.30 184.55 185.81 187.06 188.31 189.57 190.82 192.07 193.33
		     194.58 195.83 197.09 198.34 199.59 200.85 202.10 203.35 204.61
		     205.86 207.11 208.37 209.62 210.87 212.13 213.38 214.63 215.89
		     217.14 218.39 219.65 220.90 222.15 223.41 224.66 225.91 227.17
		     228.42 229.67 230.93 232.18 233.43 234.69 235.94 237.19 238.45
		     239.70 240.95 242.21 243.46 244.71 245.97 247.22 248.47 249.73
		       250.98 252.23 253.49 254.74 255.99 255.99)))
	    `(let ((table_sf (curly ,@l)))
		 (declare (type ,(format nil "const std::array<float,~a>" (length l)) table_sf)))))


	 
	 (defun init_decode_packet (packet_idx output)
	   (declare (type int packet_idx)
		    (type "std::array<std::complex<float>,65535>&" output)
		    (values int))
	   (let ((header (dot (aref ,(g `_header_data) packet_idx)
			      (data)))
		 (offset (aref ,(g `_header_offset) packet_idx))
		 (number_of_quads ,(space-packet-slot-get 'number-of-quads 'header))
		 (baq_block_length (* 8 (+ 1 ,(space-packet-slot-get 'baq-block-length 'header))))
		 
		 (number_of_baq_blocks (+ 1 (/ (* 2 number_of_quads)
					     256)))
		 (brcs)
		 (thidxs)
		 (baq_mode ,(space-packet-slot-get 'baq-mode 'header))
		 (data (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
		 
		 #+nil ("(*decoder_jump_table[5])(sequential_bit_t*)" (curly ,@(loop for i below 5 collect
										    (format nil "decode_huffman_brc~a" i))))
		 )
	     (declare #+nil (type "int" "(*decoder_jump_table[5])(sequential_bit_t*)")

		      (type ,(format nil "std::array<uint8_t,~d>"
				     (round (/ 65536 256)))
			    brcs)
		      (type ,(format nil "std::array<uint8_t,~d>"
				     (round (/ 65536 256)))
			    thidxs))
	     #-nil (do0
	      (assert (<= number_of_baq_blocks 256))
	      (assert (or ,@(loop for e in `(0 3 4 5 12 13 14) collect
				 `(== ,e baq_mode))))
	      ,(logprint "" `(packet_idx baq_mode baq_block_length)))
	     (let ((s))
	       (declare (type sequential_bit_t s))
	       (init_sequential_bit_function &s (+ (aref ,(g `_header_offset) packet_idx)
							   62 6))
	       ,@(loop for e in `(ie io qe qo)
		      collect
			(let ((sym (format nil "decoded_~a_symbols" e))
			      (sym-a (format nil "decoded_~a_symbols_a" e)))
			  `(let ((,sym 0)
				 (,sym-a))
			     (declare (type "std::array<float,65535>" ,sym-a))
			     (do0
			      ,(format nil "// parse ~a data" e)
			      (for ((= "int block" 0)
				    (< ,sym number_of_quads)
				    (incf block))
				   ,(case e
				      (ie
				       `(let ((brc (get_bit_rate_code &s)))
					  (setf (aref brcs block) brc)))
				      (io
				       `(let ((brc (aref brcs block)))))
				      (qe
				       `(let ((thidx (get_threshold_index &s))
					      (brc (aref brcs block)))
					  (setf (aref thidxs block) thidx)))
				      (qo
				       `(let ((brc (aref brcs block))
					      (thidx (aref thidxs block))))))
				   (case brc
				    ,@(loop for brc-value below 5 collect
					   `(,(format nil "case ~a" brc-value)
					      (progn
					       #+nil (do0
						      (unless (or ,@(loop for e in `(0 1 2 3 4) collect
									 `(== ,e brc)))
							,(logprint "error: out of range" `(brc)) 
							(assert 0))
						      ,(logprint (format nil "~a" e) `(brc block number_of_baq_blocks)))
					       ,(let ((th (case brc-value
								    (0 3)
								    (1 3)
								    (2 5)
								    (3 6)
								    (4 8))))
						  `(,@(if (member e `(ie io))
							  `(do0)
							  `(if (<= thidx ,th)))
						      ,@(loop for thidx-choice in (if (member e `(ie io))
										      `(thidx-unknown)
										      `(simple normal)) collect
							     `(do0
							       (for ((= "int i" 0)
								     (and (< i
									     128 ;(/ baq_block_length 2) ;; divide by two because even and odd samples are handled in different loops?
									     )
									  (< ,sym
									     number_of_quads
									     ))
								     (incf i))
								    (let ((sign_bit (get_sequential_bit &s))
									  (mcode (,(format nil "decode_huffman_brc~a" brc-value) &s))
									  (symbol_sign 1s0)
									  )
								      
								      (when sign_bit
									(setf symbol_sign -1s0))
								      (do0
								       #+nil (let ((bit s.current_bit_count)
										   (byte (static_cast<int>
											  (-  s.data
											      (static_cast<uint8_t*> ,(g `_mmap_data))
											      ))))
									       ,(logprint "" `(v ,sym i byte bit
												 block)))
								       ,(if (member e `(qe qo))
									    `(do0
									      ,(format nil "// decode ~a p.75" e)
									      ,(let ((th-mcode (case brc-value
												 (0 3)
												 (1 4)
												 (2 6)
												 (3 9)
												 (4 15))))
										 `(let ((v 0s0))
										    ,(case thidx-choice
										       (simple
											`(if (< mcode ,th-mcode)
											    (setf v (* symbol_sign mcode))
											    (setf v (* symbol_sign
												       (aref
													,(format nil "table_b~a"
														 brc-value)
													thidx)))))
										       (normal
											`(setf v (* symbol_sign
												   (aref ,(format nil
														  "table_nrl~a"
														  brc-value)
													 mcode)
												   (aref table_sf thidx))))))))
									    `(let ((v (* symbol_sign mcode)))
									       "// in ie and io we don't have thidx yet, will be processed later"))
								       
								       (setf (aref ,sym-a ,sym)
									     v)))
								    (incf ,sym))))))
					       break)))))
			      (consume_padding_bits &s)))))
	       (dotimes (block number_of_baq_blocks)
		 (let ((brc (aref brcs block))
				  (thidx (aref thidxs block)))
		  ,@(loop for e in `(ie io) collect
			 (let (	;(sym (format nil "decoded_~a_symbols" e))
			       (sym-a (format nil "decoded_~a_symbols_a" e)))
			   `(case brc
			      ,@(loop for brc-value below 5 collect
				     `(,(format nil "case ~a" brc-value)
					(progn
					  #+nil (do0
						 (unless (or ,@(loop for e in `(0 1 2 3 4) collect
								    `(== ,e brc)))
						   ,(logprint "error: out of range" `(brc)) 
						   (assert 0))
						 ,(logprint (format nil "~a" e) `(brc block number_of_baq_blocks)))
					  ,(let ((th (case brc-value
						       (0 3)
						       (1 3)
						       (2 5)
						       (3 6)
						       (4 8))))
					     `(if (<= thidx ,th)
						  ,@(loop for thidx-choice in `(simple normal) collect
							 `(do0
							   (dotimes (i 128)
							     (let ((pos (+ i (* 128 block)))
								   (scode (aref ,sym-a pos))
								   (mcode (static_cast<int> (fabsf scode)))
								   (symbol_sign (copysignf 1s0 scode)))
							       (do0
								,(format nil "// decode ~a p.75" e)
								,(let ((th-mcode (case brc-value
										   (0 3)
										   (1 4)
										   (2 6)
										   (3 9)
										   (4 15))))
								   `(let ((v 0s0))
								      ,(case thidx-choice
									 (simple
									  `(if (< mcode ,th-mcode)
									       (setf v (* symbol_sign mcode))
									       (setf v (* symbol_sign
											  (aref
											   ,(format nil "table_b~a"
												    brc-value)
											   thidx)))))
									 (normal
									  `(setf v (* symbol_sign
										      (aref ,(format nil
												     "table_nrl~a"
												     brc-value)
											    mcode)
										      (aref table_sf thidx)))))))
								(setf (aref ,sym-a pos) v))))))))
					  break))))
			   ))))

	       (dotimes (i decoded_ie_symbols)
		   (do0 (dot (aref output (* 2 i)) (real (aref decoded_ie_symbols_a i)))
			(dot (aref output (* 2 i)) (imag (aref decoded_qe_symbols_a i))))
		   (do0 (dot (aref output (+ 1 (* 2 i))) (real (aref decoded_io_symbols_a i)))
			(dot (aref output (+ 1 (* 2 i))) (imag (aref decoded_qo_symbols_a i)))))
	       
	       (let ((n (+ decoded_ie_symbols
			   decoded_io_symbols)))
		 (return n)

		 )))))))
       
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						 (merge-pathnames #P"proto2.h"
								  *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e  
	     (emit-c :code code :hook-defun 
		     #'(lambda (str)
			 (format s "~a~%" str)))
	     
	     (write-source (asdf:system-relative-pathname
			    'cl-cpp-generator2
			    (format nil
				    "~a/copernicus_~2,'0d_~a.cpp"
				    *source-dir* i name))
			   code))))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))

		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    (include <vector>
			     <array>
			     <iostream>
			     <iomanip>)
		    " "
		    (do0
		     
		    " "
		    ,@(loop for e in (reverse *utils-code*) collect
			 e)
		    ;"#define length(a) (sizeof((a))/sizeof(*(a)))"
					;"#define max(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })"
					;"#define min(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })"
		    ;"#define max(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a > _b ? _a : _b; })"
		    ;"#define min(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a < _b ? _a : _b; })"
		    		    
		    " "
		    
		    )
		    " "
		    "#endif"
		    " ")
		  )
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    
		    
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    
    
    
    ;; we need to force clang-format to always have the return type in the same line as the function: PenaltyReturnTypeOnItsOwnLine
					;(sb-ext:run-program "/bin/sh" `("gen_proto.sh"))
    #+nil (sb-ext:run-program "/usr/bin/make" `("-C" "source" "-j12" "proto2.h"))))

