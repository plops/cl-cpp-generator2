(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

;; https://docs.google.com/presentation/d/1LAm3p20egBVvj86p_gmaf-zuPYm4_OAKut5xcl9cWwk/edit?usp=sharing

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
	(ses-ssb-signal-type 0 :bits 4)
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
	    
	    (format t "~a ~a ~a ~a~%" preceding-octets preceding-bits bits default-value)
	    (if (<= bits 8)
		(let ((mask 0))
		  
		  (declare (type (unsigned-byte 8) mask))
		  (setf (ldb (byte bits (- 8 (+ bits preceding-bits))) mask) #xff)
		  (values
		   `(>> (&
			 (hex ,mask)
			 (aref ,data8 (+ 1 ,preceding-octets)))
			(- 8 (+ ,bits ,preceding-bits)))
		   'uint8_t
		   ))
		(multiple-value-bind (bytes rest-bits) (floor (+ preceding-bits bits) 8)
		  (let ((firstmask 0)
			(lastmask 0))
		    (setf (ldb (byte (- 8 preceding-bits) 0) firstmask) #xff
			  (ldb (byte rest-bits (- 8 rest-bits)) lastmask) #xff)
		    (values
		     (if (= lastmask 0)
			 `(+ 
			   ,@(loop for byte from (- bytes 1) downto 1 collect
				  `(* ,(expt 256 (- bytes byte 1))
				      (aref ,data8 ,(+ preceding-octets 0 byte))))
			   (* ,(expt 256 (- bytes 1)) (& (hex ,firstmask) (aref ,data8 ,(+ preceding-octets 0)))))
			 `(+ (>> (& (hex ,lastmask) (aref ,data8 ,(+ preceding-octets 0 bytes)))
				 ,(- 8 rest-bits))
			     ,@(loop for byte from (- bytes 1) downto 1 collect
				    `(* ,(expt 256 (- bytes byte))
					(aref ,data8 ,(+ preceding-octets 0 byte))))
			     (* ,(expt 256 bytes) (& (hex ,firstmask) (aref ,data8 ,(+ preceding-octets 0))))))
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
              "std::endl")
          )
      )
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
	       "std::endl"))
      
      )

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
		       <chrono>)
	      (let ((state ,(emit-globals :init t)))
		(declare (type State state)))
	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))
					;(vkprint "main" )
		(setf ,(g `_filename)
		      (string "/home/martin/Downloads/S1A_IW_RAW__0SDV_20191125T135230_20191125T135303_030068_036F1E_6704.SAFE/s1a-iw-raw-s-vv-20191125t135230-20191125t135303-030068-036f1e.dat"))
		(init_mmap ,(g `_filename))
		(init_collect_packet_headers)
		(init_process_packet_headers)
		(init_decode_packet 0)
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
		  <cstring>)
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
	 
	 
	 (defun init_process_packet_headers ()
	   (let ((p0 (dot (aref ,(g `_header_data) 0)
			  (data)))
		 (coarse_time0 ,(space-packet-slot-get 'coarse-time 'p0))
		 (fine_time0 (* ,(expt 2d0 -16) (+ .5 ,(space-packet-slot-get 'fine-time 'p0))))
		 (time0 (+ coarse_time0 fine_time0)))
	    (foreach (e ,(g `_header_data))
		     (let ((p (e.data))
			   (fref 37.53472224)
			   (coarse_time ,(space-packet-slot-get 'coarse-time 'p))
			   (fine_time (* ,(expt 2d0 -16) (+ .5 ,(space-packet-slot-get 'fine-time 'p))))
			   (time (- (+ coarse_time
				       fine_time)
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
			   (sync_marker ,(space-packet-slot-get 'sync-marker 'p))
			   (sync2 (+ ,@(loop for j below 4  collect
					    `(* ,(expt 256 (- 3 j)) (logand #xff (static_cast<int> (aref p (+ 12 ,j))))))))
			   (baqmod2 (logand #x1F (>> (aref p (+ 6 37)) 3)))
			   (tstmod ,(space-packet-slot-get 'test-mode 'p))
			   (swath ,(space-packet-slot-get 'ses-ssb-swath-number 'p))
			   (ele ,(space-packet-slot-get 'sab-ssb-elevation-beam-address 'p)))
		       ,(logprint "" `(time "std::hex" swst swath count pri_count rank rank2 pri baqmod sync2 sync_marker baqmod2 tstmod azi ele)))))
	   ))))

  (define-module
      `(decode_packet
	()
	(do0
	 (do0
	  ,(emit-utils :code
		       `(defstruct0 sequential_bit_t
					;(user_data_position size_t)
			  (current_bit_count size_t)
					;(current_byte size_t)
			  (data uint8_t*)
			  ))
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
			(and (>> current_byte seq_state->current_bit_count)
			     1))))
	      (when (< 7 seq_state->current_bit_count)
		(setf seq_state->current_bit_count 0)
		(incf seq_state->data))
	      (return res))))
	 (defun get_bit_rate_code (s)
	   (declare (type sequential_bit_t* s)
		    (values "inline int"))
	   "// note: evaluation order is crucial"
	   (return (+ ,@(loop for j below 3 collect
			     `(* ,(expt 2 (- 2 j))
				 (get_sequential_bit s))))))
	 (defun init_decode_packet (packet_idx)
	   (declare (type int packet_idx))
	   (let ((header (dot (aref ,(g `_header_data) packet_idx)
			      (data)))
		 (offset (aref ,(g `_header_offset) packet_idx))
		 (number_of_quads ,(space-packet-slot-get 'number-of-quads 'header))
		 (data (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
		 (baqmod ,(space-packet-slot-get 'baq-mode `header)))
	     ,(logprint "" `(packet_idx baqmod))
	     (let ((decoded_symbols 0)
		   (number_of_baq_blocks (/ (* 2 number_of_quads)
					    256))
		   (s))
	       (declare (type sequential_bit_t s))
	       (init_sequential_bit_function &s (+ (aref ,(g `_header_offset) packet_idx)
							   62 6))
	       (for ((= "int block" 0)
		     (< decoded_symbols number_of_quads)
		     ())
		    (let ((brc (get_bit_rate_code &s)))
		      ,(logprint "" `(brc))))
	       ))
	   ))))
     
   
  
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
 

