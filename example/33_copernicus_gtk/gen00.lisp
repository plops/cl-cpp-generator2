(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))

(progn
  (defparameter *source-dir* #P"example/33_copernicus_gtk/source/")
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))


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
		     (format nil "uint~a_t" (next-power-of-two bits)))))))))))

  (defun csvprint (filename &optional rest)
      `(do0
	(progn
	  (let ((outfile))
	    (declare (type "std::ofstream" outfile))
	    (outfile.open (string ,filename)
			  (logior "std::ios_base::out"
				  "std::ios_base::app"))
	    (when (== 0 (outfile.tellp))
	      (<< outfile
		  (string ,(format nil "~{~a~^,~}" (mapcar #'(lambda (x) (emit-c :code x))
						    rest)))
		  "std::endl"))
	    (<< outfile
		,@(loop for e in rest and i downfrom (1- (length rest))
		     appending
		       (if (eq i 0)
			   `(,e)
			   `(,e
			       (string ","))))
		"std::endl")
	    (outfile.close)))))
  
  (defun gen-huffman-decoder (name huffman-tree)
      (labels ((frob (tree)
	     (cond ((null tree)
		    (error "null"))
		   ((atom tree) `(return ,tree))
		   ((null (cdr tree))
		    `(return ,(car tree)))
		   (t `(if (get_sequential_bit s)
			   ,(frob (cadr tree))
			   ,(frob (car tree)))))))
       `(defun ,(format nil "decode_huffman_~a" name) (s)
	  (declare (type sequential_bit_t* s)
		   (values "inline int"))
	  ,(frob huffman-tree))))

  
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " ")
    (defparameter *global-code* nil)
    (defun emit-global (&key code)
      (push code *global-code*)
      " "))
  (progn
    
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun logprint (msg &optional rest)
      `(do0
	" "
	#-nolog
	(do0
					;("std::setprecision" 3)
	 (<< "std::cout"
	     ;;"std::endl"
	     ("std::setw" 10)
	     (dot ("std::chrono::high_resolution_clock::now")
		  (time_since_epoch)
		  (count))
					;,(g `_start_time)
	     
	     (string " ")
	     ("std::this_thread::get_id")
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
		      (string ,(format nil " ~a='" (emit-c :code e)))
		      ,e
		      (string "'")))
	     "std::endl"
	     "std::flush"))))
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  (declare (ignorable default))
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(declare (ignorable type))
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   (declare (ignorable value))
			   `(,name ,type))))))))
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	  #+nil (format t "generate ~a~%" module-name)
	  (push `(do0
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  ;(include "proto2.h")
		  " ")
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header))
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (declare (ignorable direction))
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))

  (let*  ((max-number-quads 52378))
    
    (define-module
       `(base ((_main_version :type "std::string")
	       (_code_repository :type "std::string")
	       (_code_generation_time :type "std::string")
	       (_filename :type "char const *")
	       )
	      (do0
	       
		    (include <iostream>
			     <chrono>
			     <thread>
			     <unordered_map>
			     )
		    
		    ;(include <gtkmm.h>)
		    " "

		    
		    
		    (split-header-and-code
		     (do0
		      "// header"
		      
		      (include <gtkmm-3.0/gtkmm/treeview.h>
			       <gtkmm-3.0/gtkmm/liststore.h>
			       <gtkmm-3.0/gtkmm/box.h>
			       <gtkmm-3.0/gtkmm/scrolledwindow.h>
			       <gtkmm-3.0/gtkmm/window.h>
			       ;<glibmm-2.4/glibmm/ustring.h>
			       ;<glibmm-2.4/glibmm/value_basictypes.h>
			       )
		      #+nil (include <gtkmm.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp"
			       "vis_01_mmap.hpp"
			       "vis_02_collect_packet_headers.hpp"
			       "vis_06_decode_sub_commutated_data.hpp")
		      
		      " "
		      ))

		    (let ((state ,(emit-globals :init t)))
		      (declare (type "State" state)))

		    ,(progn
		       (defun item (name members)
			 (let* (
				(class-name (format nil "CellItem_~a" name))
				(list-store-name (format nil "ListStore_~a" name))
				(class-type (format nil "~a&" class-name))
				(const-class (format nil "const ~a&" class-name)))
			   `(do0
			     (defclass ,class-name ()
			       "public:"
			       (defmethod ,class-name ()
				 (declare
				  (construct ,@(remove-if #'null
							  (loop for e in members collect
							       (destructuring-bind (var-name var-type &optional var-default) e
								 (when var-default
								   `(,(format nil "m_~a" var-name) ,var-default))))))
				  (values :constructor)))
			       (defmethod ,(format nil "~~~a" class-name) ()
				 (declare (values :constructor)))
			       (defmethod ,class-name (src)
				 (declare (values :constructor)
					  (type ,const-class src))
				 (operator= src))
			       (defmethod ,class-name (,@(loop for e in members collect
							      (destructuring-bind (var-name var-type &optional var-default) e
								var-name)))
				 (declare (values :constructor)
					  (construct ,@(loop for e in members collect
							    (destructuring-bind (var-name var-type &optional var-default) e
							      `(,(format nil "m_~a" var-name)
								 ,var-name))))
					  ,@(loop for e in members collect
						 (destructuring-bind (var-name var-type &optional var-default) e
						   `(type
						     ,var-type
						     ,var-name))))

				 (let ((p (+ offset (static_cast<uint8_t*> ,(g `_mmap_data)))))

				   ,@(loop for e in *space-packet* collect
					    (destructuring-bind (name_ default-value &key bits) e
					      (let ((cname (substitute #\_ #\- (format nil "~a" name_))))
					       `(setf ,(format nil "m_~a" cname)
						      (static_cast<int> ,(space-packet-slot-get name_ 'p))))))

				   
				   #+nil (let (
					 (fref 37.53472224d0)
					 (swst (/ (static_cast<double> ,(space-packet-slot-get 'sampling-window-start-time 'p))
						  fref))
					 (coarse_time ,(space-packet-slot-get 'coarse-time 'p))
					 (fine_time ,(space-packet-slot-get 'fine-time 'p))
					 (ftime (* ,(expt 2d0 -16) (+ .5d0 fine_time)))
					 (time0 0)
					 (time (- (+ coarse_time
						     ftime)
						  time0))
					 
					 (azi ,(space-packet-slot-get 'sab-ssb-azimuth-beam-address 'p))
					 (count ,(space-packet-slot-get 'space-packet-count 'p))
					 (pri_count ,(space-packet-slot-get 'pri-count 'p))
					 (pri (/ (static_cast<double> ,(space-packet-slot-get 'pulse-repetition-interval 'p))
						 fref))
					 (rank ,(space-packet-slot-get 'rank 'p))
					 (rank2 (static_cast<int> (aref p (+ 49))))
					 (baqmod ,(space-packet-slot-get 'baq-mode 'p))
					 (baq_n ,(space-packet-slot-get 'baq-block-length 'p))
					 (sync_marker ,(space-packet-slot-get 'sync-marker 'p))
					 (sync2 (+ ,@(loop for j below 4  collect
							  `(* ,(expt 256 (- 3 j)) (logand #xff (static_cast<int> (aref p (+ 12 ,j))))))))
					 (baqmod2 (static_cast<int> (aref p 37)) ;(logand #x1F (>> (aref p 37) 3))
					   )
					 (err ,(space-packet-slot-get 'error-flag 'p))
					 (tstmod ,(space-packet-slot-get 'test-mode 'p))
					 (rx ,(space-packet-slot-get 'rx-channel-id 'p))
					 (ecc ,(space-packet-slot-get 'ecc-number 'p))
					 (pol ,(space-packet-slot-get 'sab-ssb-polarisation 'p))
					 (signal_type ,(space-packet-slot-get 'ses-ssb-signal-type 'p))
					 (swath ,(space-packet-slot-get 'ses-ssb-swath-number 'p))
					 (ele ,(space-packet-slot-get 'sab-ssb-elevation-beam-address 'p)))
				     
				     





				     ))


				 )
			       
			       (defmethod operator= (src)
				 (declare (values ,class-type)
					  (type ,const-class src))
				 ,@(loop for e in (loop for e in members collect
						       (destructuring-bind (var-name var-type &optional var-default) e
							 (format nil "m_~a" var-name)
							 ))
				      collect
					`(setf ,e (dot src ,e)))
				 ,@(loop for e in *space-packet* collect
				      (destructuring-bind (name_ default-value &key bits) e
					(let ((cname (substitute #\_ #\- (format nil "~a" name_))))
					  `(setf ,(format nil "m_~a" cname) (dot src ,(format nil "m_~a" cname))))))
				 (return *this))
			       ,@(loop for e in members collect
				      (destructuring-bind (var-name var-type &optional var-default) e
					(format nil "~a m_~a;" var-type var-name)))
			       ,@(loop for e in *space-packet* collect
				      (destructuring-bind (name_ default-value &key bits) e
					(let ((cname (substitute #\_ #\- (format nil "~a" name_))))
					  (format nil "gint m_~a;" cname))))
			       )
			     (defclass ,list-store-name "public Gtk::Window"
			       "public:"
			       (defmethod ,list-store-name ()
				 (declare (values :constructor)
					  (construct (m_VBox Gtk--ORIENTATION_VERTICAL 8)
						     (m_Label (string ,list-store-name))))
				 (set_title (string ,(format nil "Gtk::ListStore ~a" list-store-name)))
				 (set_border_width 8)
				 (set_default_size 280 250)
				 (add m_VBox)
				 (m_VBox.pack_start m_Label Gtk--PACK_SHRINK)
				 (m_ScrolledWindow.set_shadow_type Gtk--SHADOW_ETCHED_IN)
				 (m_ScrolledWindow.set_policy
				  ; Gtk--POLICY_NEVER
				  Gtk--POLICY_AUTOMATIC
				  Gtk--POLICY_AUTOMATIC)
				 (m_VBox.pack_start m_ScrolledWindow)
				 (create_model)
				 (m_TreeView.set_model m_refListStore)
				 (m_TreeView.set_search_column (m_columns.offset.index))
				 (m_TreeView.set_has_tooltip true)
				 (dot m_TreeView
				      (signal_query_tooltip)
				      (connect (lambda (x y keyboard_tooltip tooltip)
						 (declare (type int x y)
							  (capture this)
							  (type bool keyboard_tooltip)
							  (type "const Glib::RefPtr<Gtk::Tooltip>&" tooltip)
							  (values bool))
						 (tooltip->set_text (string "hello"))
						 (if keyboard_tooltip
						     (do0
						      
						      (let ((path)
							    (focus_column)
							    )
							(declare (type "Gtk::TreeModel::Path" path)
								 (type "Gtk::TreeViewColumn*" focus_column))
							(m_TreeView.get_cursor path focus_column)
							,(logprint "keyboard" `(path focus_column))
							))
						     (do0
						      
						      (let ((bx 0)
							    (by 0))
							(declare (type int bx by))
							(m_TreeView.convert_widget_to_bin_window_coords x y bx by)
							(let ((path)
							      (column)
							      (cx 0)
							      (cy 0)
							      )
							  (declare (type "Gtk::TreeModel::Path" path)
								   (type "Gtk::TreeViewColumn*" column)
								   (type int cx cy))
							  (m_TreeView.get_path_at_pos bx by path column cx cy)
							  
							  ,(logprint "mouse" `(x y bx by cx cy
										 (path.to_string)
										 (? (== nullptr (path.begin))
										    -1
										    (aref (path.begin) 0))
										 column
										 (column->get_title)
										 ;(column->get_sort_column_id)
										 ))))))
						 (return true))))
				 (add_columns)
				 (m_ScrolledWindow.add m_TreeView)
				 (show_all)
				 )
			       (defmethod ,(format nil "~~~a" list-store-name) ()
				 (declare (values :constructor)))
			       "protected:"
			       (defmethod create_model ()
				 (declare (virtual))
				 (setf m_refListStore (Gtk--ListStore--create m_columns))
				 (add_items)
				 (std--for_each
				  (m_vecItems.begin)
				  (m_vecItems.end)
				  (sigc--mem_fun *this
						 ,(format nil "&~a::liststore_add_item" list-store-name))))
			       (defmethod add_columns ()
				 (declare (virtual))
				 #+nil (let ((cols_count (m_TreeView.append_column_editable (string "offset")
											    m_columns.offset))
					     (pColumn (m_TreeView.get_column (- cols_count 1))))
					 ;; set to fixed 50 pixel size
					 (pColumn->set_sizing Gtk--TREE_VIEW_COLUMN_FIXED)
					 (pColumn->set_fixed_width 60)
					 (pColumn->set_clickable))
				 #+nil (m_TreeView.append_column (string "Bug Number")
								 m_columns.number)
				 ,@(loop for e in members collect
					(destructuring-bind (var-name var-type &optional var-default) e
					  `(do0
					    (m_TreeView.append_column
					     (string ,var-name)
					     (dot m_columns ,var-name))
					    ;(m_TreeView.set_tooltip_text (string ,var-name))
					    )))
				 ,@(loop for e in *space-packet* collect
				      (destructuring-bind (name_ default-value &key bits) e
					(let ((cname (substitute #\_ #\- (format nil "~a" name_)))
					      (short-name
					       (let* ((s (format nil "~a" name_))
						     (l (list (aref s 0)))
						     (add-the-next nil))
						 (loop for c across (subseq s 1)
						    do
						      (if (eq c #\-)
							  (setf add-the-next t)
							  (when add-the-next
							    (setf add-the-next nil)
							    (push c l)))
						      )
						 (format nil "~{~a~}" (reverse l)))))
					  `(do0
					    (m_TreeView.append_column
					     (string ,short-name)
					     (dot m_columns ,cname))
					    #+nil (m_TreeView.set_tooltip_cell (string ,name_)
									 nullptr
									 2
									 nullptr))
					  ))))
			       (defmethod add_items ()
				 (declare (virtual))
				 (for-range (val ,(g `_header_offset))
					    (dot m_vecItems
						 (push_back (,class-name val #+nil ,@(loop for e in members collect
											  (destructuring-bind (var-name var-type &optional var-default) e
											    var-name))))))
				 
				 #+nil
				 ,@(loop for e in `((false 60482 Normal "scrollable notebuooks")
						    (false 60539 Major "trisatin"))
				      collect
					(destructuring-bind (b n type str) e
					  `(dot m_vecItems
						(push_back (CellItem_Bug ,b ,n (string ,type) (string ,str)))))))
			       (defmethod liststore_add_item (foo)
				 (declare (virtual)
					  (type ,const-class foo))
				 (let ((row (deref (m_refListStore->append))))
				   ,@(loop for e in members collect
					  (destructuring-bind (var-name var-type &optional var-default) e
					    `(setf (aref row (dot m_columns ,var-name))
						   (dot foo ,(format nil "m_~a" var-name)))))
				   ,@(loop for e in *space-packet* collect
					  (destructuring-bind (name_ default-value &key bits) e
					    (let ((cname (substitute #\_ #\- (format nil "~a" name_)))
						  )
					      `(setf (aref row (dot m_columns ,cname))
						     (dot foo ,(format nil "m_~a" cname)))
					      
						)))
				   
				   #+nil,@(loop for e in `(fixed number severity description) collect
					       `(setf (aref row (dot m_columns ,e))
						      (dot foo ,(format nil "m_~a" e))))))
			       "Gtk::Box m_VBox;"
			       "Gtk::ScrolledWindow m_ScrolledWindow;"
			       "Gtk::Label m_Label;"
			       "Gtk::TreeView m_TreeView;"
			       "Glib::RefPtr<Gtk::ListStore> m_refListStore;"

			       ,(format nil "typedef std::vector<~a> type_vecItems;" class-name)
			       "type_vecItems m_vecItems;"
			       (do0
				(space "struct ModelColumns : public Gtk::TreeModelColumnRecord"
				       (progn
					 ,@(loop for e in members collect
						(destructuring-bind (var-name var-type &optional var-default) e
						  (format nil "Gtk::TreeModelColumn<~a> ~a;" var-type var-name)))
					 ,@(loop for e in *space-packet* collect
					    (destructuring-bind (name_ default-value &key bits) e
					      (let ((cname (substitute #\_ #\- (format nil "~a" name_))))
						(format nil "Gtk::TreeModelColumn<gint> ~a;" cname)
						)))
					 (defun+ ModelColumns ()
					   (declare (values :constructor))
					   ,@(loop for e in members collect
						  (destructuring-bind (var-name var-type &optional var-default) e
						    `(add ,var-name)))
					   ,@(loop for e in *space-packet* collect
					    (destructuring-bind (name_ default-value &key bits) e
					      (let ((cname (substitute #\_ #\- (format nil "~a" name_))))
						`(add ,cname)
						))))))
				
				"const ModelColumns m_columns;")))))
		       
		       (item "SpacePacketHeader0" `((offset gsize 0))))

		    #+nil (do0
		     (defclass CellItem_Bug ()
		       "public:"
		       (defmethod CellItem_Bug ()
			 (declare
			  (construct (m_fixed false)
				     (m_number 0))
			  (values :constructor)))
		       (defmethod ~CellItem_Bug ()
			 (declare (values :constructor)))
		       (defmethod CellItem_Bug (src)
			 (declare (values :constructor)
				  (type "const CellItem_Bug&" src))
			 (operator= src))
		       (defmethod CellItem_Bug (fixed number severity description)
			 (declare (values :constructor)
				  (construct (m_fixed fixed)
					     (m_number number)
					     (m_severity severity)
					     (m_description description))
				  (type bool fixed)
				  (type guint number)
				  (type "const Glib::ustring&" severity)
				  (type "const Glib::ustring&" description)))
		       (defmethod operator= (src)
			 (declare (values CellItem_Bug&)
				  (type "const CellItem_Bug&" src))
			 ,@(loop for e in `(m_fixed m_number m_severity m_description) collect
				`(setf ,e (dot src ,e)))
			 (return *this))
		       "bool m_fixed;"
		       "guint m_number;"
		       "Glib::ustring m_severity;"
		       "Glib::ustring m_description;")


		     (defclass Example_TreeView_ListStore "public Gtk::Window"
		       "public:"
		       (defmethod Example_TreeView_ListStore ()
			 (declare (values :constructor)
				  (construct (m_VBox Gtk--ORIENTATION_VERTICAL 8)
					     (m_Label (string "This is the bug list."))))
			 (set_title (string "Gtk::ListStore demo"))
			 (set_border_width 8)
			 (set_default_size 280 250)
			 (add m_VBox)
			 (m_VBox.pack_start m_Label Gtk--PACK_SHRINK)
			 (m_ScrolledWindow.set_shadow_type Gtk--SHADOW_ETCHED_IN)
			 (m_ScrolledWindow.set_policy Gtk--POLICY_NEVER Gtk--POLICY_AUTOMATIC)
			 (m_VBox.pack_start m_ScrolledWindow)
			 (create_model)
			 (m_TreeView.set_model m_refListStore)
			 (m_TreeView.set_search_column (m_columns.description.index))
			 (add_columns)
			 (m_ScrolledWindow.add m_TreeView)
			 (show_all)
			 )
		       (defmethod ~Example_TreeView_ListStore ()
			 (declare (values :constructor)
				  ;; override
				  ))
		       "protected:"
		       (defmethod create_model ()
			 (declare (virtual))
			 (setf m_refListStore (Gtk--ListStore--create m_columns))
			 (add_items)
			 (std--for_each
			  (m_vecItems.begin)
			  (m_vecItems.end)
			  (sigc--mem_fun *this
					 &Example_TreeView_ListStore--liststore_add_item)))
		       (defmethod add_columns ()
			 (declare (virtual))
			 (let ((cols_count (m_TreeView.append_column_editable (string "Fixed?")
									      m_columns.fixed))
			       (pColumn (m_TreeView.get_column (- cols_count 1))))
			   ;; set to fixed 50 pixel size
			   (pColumn->set_sizing Gtk--TREE_VIEW_COLUMN_FIXED)
			   (pColumn->set_fixed_width 60)
			   (pColumn->set_clickable))
			 (m_TreeView.append_column (string "Bug Number")
						   m_columns.number)
			 (m_TreeView.append_column (string "Severity")
						   m_columns.severity)
			 (m_TreeView.append_column (string "Description")
						   m_columns.description)
			 )
		       (defmethod add_items ()
			 (declare (virtual))
			 ,@(loop for e in `((false 60482 Normal "scrollable notebuooks")
					    (false 60539 Major "trisatin"))
			      collect
				(destructuring-bind (b n type str) e
				  `(dot m_vecItems
					(push_back (CellItem_Bug ,b ,n (string ,type) (string ,str)))))))
		       (defmethod liststore_add_item (foo)
			 (declare (virtual)
				  (type "const CellItem_Bug&" foo))
			 (let ((row (deref (m_refListStore->append))))
			   ,@(loop for e in `(fixed number severity description) collect
				  `(setf (aref row (dot m_columns ,e))
					 (dot foo ,(format nil "m_~a" e))))))
		       "Gtk::Box m_VBox;"
		       "Gtk::ScrolledWindow m_ScrolledWindow;"
		       "Gtk::Label m_Label;"
		       "Gtk::TreeView m_TreeView;"
		       "Glib::RefPtr<Gtk::ListStore> m_refListStore;"

		       "typedef std::vector<CellItem_Bug> type_vecItems;"
		       "type_vecItems m_vecItems;"
		       (do0
			,(let ((l `((bool fixed)
				    ("unsigned int" number)
				    ("Glib::ustring" severity)
				    ("Glib::ustring" description))))
			   `(space "struct ModelColumns : public Gtk::TreeModelColumnRecord"
				   (progn
				     ,@(loop for (e f) in l collect
					    (format nil "Gtk::TreeModelColumn<~a> ~a;" e f))
				     (defun+ ModelColumns ()
				       (declare (values :constructor))
				       ,@(loop for (e f) in l collect
					      `(add ,f))))))
			"const ModelColumns m_columns;")
		       
		       ))

		    

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))

		      (setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))

		      (setf ,(g `_filename)
		      (string
		       "/media/sdb4/sar/sao_paulo/s1b-s6-raw-s-vv-20200824t214314-20200824t214345-023070-02bce0.dat"
		       ))
		      (init_mmap ,(g `_filename))
		      (init_collect_packet_headers)

		      
		      #+nil (do0
		       (let ((packet_idx 0)
		       (map_ele)
		       (map_cal)
		       (map_sig)
		       (cal_count 0))
		   (declare (type "std::unordered_map<int,int>" map_ele map_cal map_sig))
		   (init_sub_commutated_data_decoder)
		   (remove (string  "./o_anxillary.csv"))
		   (foreach (e ,(g `_header_data))
			    (let ((offset (aref ,(g `_header_offset) packet_idx))
				  (p (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
				  (cal_p ,(space-packet-slot-get 'sab-ssb-calibration-p 'p) )
				  (ele ,(space-packet-slot-get 'sab-ssb-elevation-beam-address 'p))
				  (cal_type (logand ele #x7))
				  (number_of_quads ,(space-packet-slot-get 'number-of-quads 'p))
				  (baq_mode ,(space-packet-slot-get 'baq-mode 'p))
				  (test_mode ,(space-packet-slot-get 'test-mode 'p))
				  (space_packet_count ,(space-packet-slot-get 'space-packet-count 'p))
				  (sub_index ,(space-packet-slot-get 'sub-commutated-index 'p))
				  (sub_data ,(space-packet-slot-get 'sub-commutated-data 'p))
				  (signal_type ,(space-packet-slot-get 'ses-ssb-signal-type 'p)))
			      
			      
			      (feed_sub_commutated_data_decoder (static_cast<uint16_t> sub_data) sub_index space_packet_count)
			      (incf (aref map_sig signal_type))
			      (if cal_p
				  (do0
				   (incf cal_count)
				   (incf (aref map_cal
					       (logand ele #x7)))
				   ,(logprint "cal" `(cal_p cal_type number_of_quads baq_mode test_mode)))
				  (do0
				   (incf (aref map_ele ele) number_of_quads)))
			      (incf packet_idx)))

		   (foreach (cal map_cal)
			    (let ((number_of_cal cal.second)
				  (cal_type cal.first))
			      ,(logprint "map_ele" `(cal_type number_of_cal))))
		   (foreach (sig map_sig)
			    (let ((number_of_sig sig.second)
				  (sig_type sig.first))
			      ,(logprint "map_sig" `(sig_type number_of_sig))))
		   

		   (let ((ma -1s0)
			 (ma_ele -1))
		     #-nil (foreach (elevation map_ele)
			      (let ((number_of_Mquads (/ (static_cast<float> elevation.second) 1e6))
				    (elevation_beam_address elevation.first))
				(when (< ma number_of_Mquads)
				  (setf ma number_of_Mquads
					ma_ele elevation_beam_address))
				,(logprint "map_ele" `(elevation_beam_address number_of_Mquads))))
		     ,(logprint "largest ele" `(ma_ele ma cal_count)))


		   
		   (let ((mi_data_delay 10000000) ;; minimum number of samples before data is stored
			 (ma_data_delay -1) ;; maximum number of samples before data is stored
			 (ma_data_end -1) ;; maximum number of samples
			 ;; earliest echos are ma_data_delay-mi_data_delay samples earlier than the latest
			 ;; at most we may need to store ma_data_delay - mi_data_delay + ma_data_end samples
			 (ele_number_echoes 0))
		    (progn
		      (let ((map_azi)
			    (packet_idx2 0))
			(declare (type "std::unordered_map<int,int>" map_azi))
			(foreach (e ,(g `_header_data))
				 (let ((offset (aref ,(g `_header_offset) packet_idx2))
				       (p (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
				       (ele ,(space-packet-slot-get 'sab-ssb-elevation-beam-address 'p))
				       (azi ,(space-packet-slot-get 'sab-ssb-azimuth-beam-address 'p))
				       (number_of_quads ,(space-packet-slot-get 'number-of-quads 'p))
				       (cal_p ,(space-packet-slot-get 'sab-ssb-calibration-p 'p) )
				       (data_delay (+ ,(/ 320 8)
						      ,(space-packet-slot-get 'sampling-window-start-time 'p))))
				   (unless cal_p
				    (when (== ele ma_ele)
				      (incf ele_number_echoes)
				      (when (< data_delay mi_data_delay)
					(setf mi_data_delay data_delay))
				      (when (< ma_data_delay data_delay )
					(setf ma_data_delay data_delay))
				      (let ((v (+ data_delay (* 2 number_of_quads))))
					(when (< ma_data_end v)
					  (setf ma_data_end v)))
				      (incf (aref map_azi azi) number_of_quads)))
				   (incf packet_idx2)))
			,(logprint "data_delay" `(mi_data_delay ma_data_delay ma_data_end ele_number_echoes))
			(foreach (azi map_azi)
				 (let ((number_of_Mquads (/ (static_cast<float> azi.second) 1e6))
				       (azi_beam_address azi.first))
				   ,(logprint "map_azi" `(azi_beam_address number_of_Mquads)))))))))

		      
		      
		      (let ((app (Gtk--Application--create argc argv
							   (string "org.gtkmm.example")))
			    (hw))
			(declare (type ListStore_SpacePacketHeader0 ;Example_TreeView_ListStore
				       hw))
			(app->run hw)))

		    
		    )))
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
		  <iostream>
		  <thread>)
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
		     ,(g `_mmap_data) data)))))))
    (define-module
      `(collect_packet_headers
	(;(_header_data :direction 'out :type void*)
	 (_header_data :direction 'out :type "std::vector<std::array<uint8_t,62+6>>")
	 (_header_offset :direction 'out :type "std::vector<size_t>"))
	(do0
	 (include <array>
		  <iostream>
		  <vector>
		  <cstring>
		  <thread>)
	 #+nil (defstruct0 space_packet_header_info_t
	   (head "std::array<uint8_t, 62+6>")
	   (offset size_t))
	 (defun destroy_collect_packet_headers ())
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
		 (incf offset (+ 6 1 data_length)))))))))

    (define-module
      `(process_packet_headers
	()
	(do0
	 (include <unistd.h>)
	 (defun init_process_packet_headers ()
	   ;(declare (values void))
	   (let ((p0 (dot (aref ,(g `_header_data) 0)
			  (data)))
		 (coarse_time0 ,(space-packet-slot-get 'coarse-time 'p0))
		 (fine_time0 (* ,(expt 2d0 -16) (+ .5d0 ,(space-packet-slot-get 'fine-time 'p0))))
		 (time0 (+ coarse_time0 fine_time0))
		 (packet_idx 0))
	     (foreach (e ,(g `_header_data))
		      (let ((offset (aref ,(g `_header_offset) packet_idx))
			      (p (+ offset (static_cast<uint8_t*> ,(g `_mmap_data)))))
			  (incf packet_idx))
		     (let (;(p (e.data))
			   (fref 37.53472224d0)
			   (swst (/ (static_cast<double> ,(space-packet-slot-get 'sampling-window-start-time 'p))
				    fref))
			   (coarse_time ,(space-packet-slot-get 'coarse-time 'p))
			   (fine_time ,(space-packet-slot-get 'fine-time 'p))
			   (ftime (* ,(expt 2d0 -16) (+ .5d0 fine_time)))
			   (time (- (+ coarse_time
				       ftime)
				    time0))
			   
			   (azi ,(space-packet-slot-get 'sab-ssb-azimuth-beam-address 'p))
			   (count ,(space-packet-slot-get 'space-packet-count 'p))
			   (pri_count ,(space-packet-slot-get 'pri-count 'p))
			   (pri (/ (static_cast<double> ,(space-packet-slot-get 'pulse-repetition-interval 'p))
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
			   "std::flush"))))))))


    (define-module
       `(decode_packet
	 ()
	 (do0
	  (do0
	   (include <cassert>
		    <cmath>
		    <thread>)
	   ,(emit-utils :code
			`(do0
			  (include <complex>)
			  ,(format nil "enum{MAX_NUMBER_QUADS=~a}; // page 55" max-number-quads) 
			  (defstruct0 sequential_bit_t
					;(user_data_position size_t)
			      (current_bit_count size_t)
					;(current_byte size_t)
			    (data uint8_t*))))
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
		   seq_state->current_bit_count 0)
	     #+log-consume ,(logprint "start sequential bit function" `((- seq_state->data
								     (static_cast<uint8_t*> ,(g `_mmap_data)))
									seq_state->current_bit_count)))

	   ,(emit-utils
	    :code
	    `(do0
	      ;(include "globals.h")
	      ;"extern State state;"
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
		 (return res)))
	      (defun get_threshold_index (s)
	    (declare (type sequential_bit_t* s)
		     (values "inline uint8_t"))
	    (return (static_cast<uint8_t> (+ ,@(loop for j below 8 collect
			       `(* (hex ,(expt 2 (- 7 j)))
				   (get_sequential_bit s)))))))))


	   (defun consume_padding_bits (s)
	    (declare (type sequential_bit_t* s)
		     (values "void"))
	    ;; fixme: mod on current_bit_count is not working
	    ;; check for odd/even byte instead
	    (let ((byte_offset (static_cast<int> (- s->data
						    (static_cast<uint8_t*> ,(g `_mmap_data))))))
	      "// make sure we are at first bit of an even byte in the next read"
	      
	      
	      (if (== 0 (% byte_offset 2))
		  (do0
		   "// we are in an even byte"
		   
		   (if (== 0 s->current_bit_count)
		       (do0
			"// nothing to be done"
			#+log-consume
			,(logprint "start consume from even byte on border, do nothing" `(byte_offset s->current_bit_count)))
		       (do0
			#+log-consume
			,(logprint "start consume from even byte" `(byte_offset s->current_bit_count))
			(incf s->data 2)
			(setf s->current_bit_count 0))))
		  (do0
		   "// we are in an odd byte"
		   #+log-consume
		   ,(logprint "start consume from odd byte" `(byte_offset s->current_bit_count))
		   (incf s->data 1)
		   (setf s->current_bit_count 0)))
	      ;#+log-consume
	      #+log-consume ,(logprint "after consume" `((- s->data
						      (static_cast<uint8_t*> ,(g `_mmap_data)))
						   s->current_bit_count
						   )))
	    #+nil
	    (let ((pad (- 16
			  (% s->current_bit_count 16))))
	      ,(logprint "" `(pad))
	      (dotimes (i pad)
		(assert (== 0
			    (get_sequential_bit s))))))
	   
	   (defun get_bit_rate_code (s)
	    (declare (type sequential_bit_t* s)
		     (values "inline uint8_t"))
	    "// note: evaluation order is crucial"
	    #+nil(let ((a (get_sequential_bit s))
		       (b (get_sequential_bit s))
		       (c (get_sequential_bit s))))
	    #+nil (<<
		   "std::cout"
		   ,@ (let ((bits (destructuring-bind (name default &key bits) (find name_  *space-packet*    :key #'first)
				    bits)))
			(loop for j from (1- bits) downto 0 collect
			     `(static_cast<int> (logand 1 (>> v ,j)))))
		   "std::endl")
	    (let ((brc (static_cast<uint8_t> (+ ,@(loop for j below 3 collect
				   `(* (hex ,(expt 2 (- 2 j)))
				       (get_sequential_bit s)))))))
	      #+safety
	      (unless (or ,@(loop for e below 5 collect `(== ,e brc)))
		,(logprint "brc out of range" `(s->current_bit_count
						(- s->data (static_cast<uint8_t*> ,(g `_mmap_data)))
						brc))
		(throw ("std::out_of_range" (string "brc")))
		;(assert 0)
		)
	      (return brc)))
	   )

	  
	  
	  

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
		       (declare (type ,(format nil "const std::array<const float,~a>" (length l)) ,table))))))

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
		       (declare (type ,(format nil "const std::array<const float,~a>" (length l)) ,table))))))
	  
	  
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
		 (declare (type ,(format nil "extern const std::array<const float,~a>" (length l)) table_sf)))))

	  

	  
	  (defun init_decode_packet (packet_idx ; mi_data_delay
				     output
				     )
	    (declare (type int packet_idx ;mi_data_delay
			   )
		     #+nil (type "std::array<std::complex<float>,MAX_NUMBER_QUADS>&" output)
		     (type "std::complex<float>*" output)
		     (values int))

	    "// packet_idx .. index of space packet 0 .."
	    "// mi_data_delay .. if -1, ignore; otherwise it is assumed to be the smallest delay in samples between tx pulse start and data acquisition and will be used to compute a sample offset in output so that all echos of one sar image are aligned to the same time offset"
	    "// output .. array of complex numbers"
	    "// return value: number of complex data samples written"
	    
	    (let ((header (dot (aref ,(g `_header_data) packet_idx)
			       (data)))
		  (offset (aref ,(g `_header_offset) packet_idx))
		  (number_of_quads ,(space-packet-slot-get 'number-of-quads 'header))
		  (baq_block_length (* 8 (+ 1 ,(space-packet-slot-get 'baq-block-length 'header))))
		  
		  (number_of_baq_blocks (static_cast<int> (round (ceil (/ (static_cast<double> (* 2d0 number_of_quads))
							 256)))))
		  (brcs)
		  (thidxs)
		  (baq_mode ,(space-packet-slot-get 'baq-mode 'header))
		  (fref 37.53472224d0)
		  (swst (/ (static_cast<double> ,(space-packet-slot-get 'sampling-window-start-time 'header))
			   fref))
		  (delta_t_suppressed (/ 320d0 (* 8 fref)))
		  (data_delay_us (+ swst delta_t_suppressed))
		  (data_delay (+ ,(/ 320 8)
				 ,(space-packet-slot-get 'sampling-window-start-time 'header)))
		  #+nil (data_offset (- data_delay mi_data_delay))
			   
		  (data (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
		  
		  #+nil ("(*decoder_jump_table[5])(sequential_bit_t*)" (curly ,@(loop for i below 5 collect
										     (format nil "decode_huffman_brc~a" i))))
		  )
	      (declare #+nil (type "int" "(*decoder_jump_table[5])(sequential_bit_t*)")

		       (type ,(format nil "std::array<uint8_t,~d>"
				      (round (/ max-number-quads 256)))
			     brcs)
		       (type ,(format nil "std::array<uint8_t,~d>"
				      (round (/ max-number-quads 256)))
			     thidxs))

	      #+nil (when (== -1 mi_data_delay)
		(setf data_offset 0))
	      #+safety (do0
			#+nil (assert (or (== -1 mi_data_delay)
				    (<= mi_data_delay data_delay)))
			(assert (<= number_of_baq_blocks 256))
			(assert (or ,@(loop for e in `(0 3 4 5 12 13 14) collect
					   `(== ,e baq_mode))))
			#+log-brc ,(logprint "" `(packet_idx baq_mode baq_block_length
						   data_delay_us
						   data_delay
						   number_of_quads
						   )))
	      (let ((s))
		(declare (type sequential_bit_t s))
		(init_sequential_bit_function &s (+ (aref ,(g `_header_offset) packet_idx)
						    62 6))
		,@(loop for e in `(ie io qe qo)
		     collect
		       (let ((sym
			      (format nil "decoded_~a_symbols" e))
			     (sym-a (format nil "decoded_~a_symbols_a" e)))
			 `(let ((,sym 0)
				(,sym-a))
			    (declare (type "std::array<float,MAX_NUMBER_QUADS>" ,sym-a))
			    (do0
			     (dotimes (i MAX_NUMBER_QUADS)
				  (setf (aref ,sym-a i) 0s0)))
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
					   `(,(format nil "~a" brc-value)
					      (progn
						#+safety (do0
							  #+nil (unless (or ,@(loop for e in `(0 1 2 3 4) collect
									     `(== ,e brc)))
							    ,(logprint "error: out of range" `(brc)) 
							    (assert 0))
					
							  #+log-brc ,(logprint (format nil "~a" e) `((static_cast<int> brc) block number_of_baq_blocks ,(if (member e `(qe qo))
																 `(static_cast<int> thidx)
																 1)))
							  )
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
								,(format nil "// reconstruction law block=~a thidx-choice=~a brc=~a" e
									 thidx-choice brc-value)
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
									   (mcode_f (static_cast<float> mcode))
									   (symbol_sign 1s0)
									   )
								       #+nil ,(logprint (format nil "huff brc=~a block=~a"
											  brc-value e) `(mcode))
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
											 `(handler-case
											     (do0
											      (if (< mcode ,th-mcode)
												   (setf v (* symbol_sign mcode_f))
												   (if (== mcode ,th-mcode)
												       (setf v (* symbol_sign
														  (dot
														   ,(format nil "table_b~a"
															    brc-value)
														   (at thidx))))
												       (do0
													,(logprint "mcode too large" `(mcode))
													(assert 0)))))
											   ("std::out_of_range&" (e)
											     ,(logprint
											       (format nil "exception simple brc=~a"
												       brc-value)
													`(thidx packet_idx))
											     (assert 0))
											   ))
											(normal
											 `(handler-case
											  (do0 (setf v (* symbol_sign
													  (dot ,(format nil
															"table_nrl~a"
															brc-value)
													       (at mcode))
													  (dot table_sf (at thidx)))))
											   ("std::out_of_range&" (e)
											     ,(logprint
											       (format nil "exception normal nrl or sf brc=~a"
												       brc-value)
													`(thidx packet_idx))
											     (assert 0))))))))
									     `(let ((v (* symbol_sign mcode_f)))
										"// in ie and io we don't have thidx yet, will be processed later"))
									
									(setf (aref ,sym-a ,sym)
									      v)))
								     (incf ,sym))))))
						break)))
				    #+safety
				    (t (progn
					 #-nolog ,(logprint "error brc out of range" `(brc))
					 (assert 0)
					;(throw ("std::out_of_range" (string "brc")))
					 break))))
			     (consume_padding_bits &s)))))
		;,(logprint "decode ie and io blocks" `(number_of_baq_blocks))
		,@(loop for e in `(ie io) collect
		 `(dotimes (block number_of_baq_blocks)
		   (let ((brc (aref brcs block))
			 (thidx (aref thidxs block)))
		     (do0
			    ,(let ( ;(sym (format nil "decoded_~a_symbols" e))
				  (sym-a (format nil "decoded_~a_symbols_a" e)))
			      `(case brc
				 ,@(loop for brc-value below 5 collect
					`(,(format nil " ~a" brc-value)
					   (progn
					     #+safety (do0
						       #+nil (unless (or ,@(loop for e in `(0 1 2 3 4) collect
										`(== ,e brc)))
							       ,(logprint "error: out of range" `(brc)) 
							       (assert 0))
						       #+nil ,(logprint (format nil "~a" e) `((static_cast<int> brc) block number_of_baq_blocks))
						       )
					     ,(format nil "// decode ~a p.74 reconstruction law middle choice brc=~a" e
						      brc-value)
					     ,(let ((th (case brc-value
							  (0 3)
							  (1 3)
							  (2 5)
							  (3 6)
							  (4 8))))
						`(if (<= thidx ,th)
						     ,@(loop for thidx-choice in `(simple normal) collect
							    `(do0
							      ,(format nil "// decode ~a p.74 reconstruction law ~a brc=~a"
								       e thidx-choice brc-value)
							      (for ((= "int i" 0)
								    (and (< i 128)
									 (< (+ i (* 128 block))
									    ,(format nil "decoded_~a_symbols" e)))
								    (incf i)) ;dotimes (i 128)
								   (let ((pos
									  (+ i (* 128 block)))
									 (scode (aref ,sym-a pos))
									 (mcode_f (fabsf scode))
									 (mcode (static_cast<int> mcode_f))
									 
								      (symbol_sign (copysignf 1s0 scode)))
								  (do0
								   ,(format nil "// decode ~a p.74 reconstruction law right side" e)
								   ,(let ((th-mcode (case brc-value
										      (0 3)
										      (1 4)
										      (2 6)
										      (3 9)
										      (4 15))))
								      `(let ((v 0s0))
									 ,(case thidx-choice
									    (simple
									     `
									     (handler-case
									      (do0 (if (< mcode ,th-mcode)
										       (setf v (* symbol_sign mcode_f))
										       (if (== mcode ,th-mcode)
											   (setf v (* symbol_sign
												      (dot
												       ,(format nil "table_b~a"
														brc-value)
												       (at thidx))))
											   (do0
											    #-nolog ,(logprint "mcode too large" `(mcode))
											    (assert 0)))))
									       ("std::out_of_range&" (e)
											     ,(logprint
											       (format nil "exception simple block=~a brc=~a"
												       e brc-value)
											       `((static_cast<int> thidx) mcode packet_idx))
											     (assert 0))))
									    (normal
									     `
									     (handler-case
									      (do0 (setf v (* symbol_sign
											      (dot ,(format nil
													    "table_nrl~a"
													    brc-value)
												   (at mcode))
											      (dot table_sf (at thidx)))))
									       ("std::out_of_range&" (e)
											     ,(logprint
											       (format nil "exception normal nrl or sf block=~a brc=~a"
												       e brc-value)
											       `((static_cast<int> thidx) block i
												 mcode packet_idx pos scode symbol_sign
												 
												 ,(format nil "decoded_~a_symbols" e)
												 ))
											     (assert 0)))))))
								   (setf (aref ,sym-a pos) v))))))))
					     break)))
				 #+safety
				 (t (progn
				      ,(logprint "unknown brc" `((static_cast<int> brc)))
				      (assert 0)
				      break))))))))
		(do0
		 (assert (== decoded_ie_symbols
			     decoded_io_symbols 
			     ))
		 (assert (== decoded_ie_symbols
			     decoded_qe_symbols
			     ))
		 (assert (== decoded_qo_symbols
			     decoded_qe_symbols
			     )))
		(dotimes (i decoded_ie_symbols)
		  (do0 (dot (aref output (* 2 i)) (real (aref decoded_ie_symbols_a i)))
		       (dot (aref output (* 2 i)) (imag (aref decoded_qe_symbols_a i))))
		  (do0 (dot (aref output (+ 1 (* 2 i))) (real (aref decoded_io_symbols_a i)))
		       (dot (aref output (+ 1 (* 2 i))) (imag (aref decoded_qo_symbols_a i)))))
		(let ((n (+ decoded_ie_symbols
			    decoded_io_symbols) ))
		  (return n)))
	      )))))

    (define-module
       `(decode_type_ab_packet
	 ()
	 (do0
	  (do0
	   (include <cassert>)
	   (include "vis_04_decode_packet.hpp")
	   (defun get_data_type_a_or_b (s)
	     (declare (type sequential_bit_t* s)
		      (values "inline int"))
	     (return (+ ,@(loop for j below 10 collect
			      `(* (hex ,(expt 2 (- 9 j)))
				  (get_sequential_bit s)))))))
	  (defun init_decode_packet_type_a_or_b
	      (packet_idx ; mi_data_delay
	       output)
	    (declare (type int packet_idx ;mi_data_delay
			   )
		     #+nil (type "std::array<std::complex<float>,MAX_NUMBER_QUADS>&" output)
		     (type "std::complex<float>*" output)
		     (values int))
	    "// packet_idx .. index of space packet 0 .."
	    "// mi_data_delay .. if -1, ignore; otherwise it is assumed to be the smallest delay in samples between tx pulse start and data acquisition and will be used to compute a sample offset in output so that all echos of one sar image are aligned to the same time offset"
	    "// output .. array of complex numbers"
	    "// return value: number of complex data samples written"
	    (let ((header (dot (aref ,(g `_header_data) packet_idx)
			       (data)))
		  (offset (aref ,(g `_header_offset) packet_idx))
		  (number_of_quads ,(space-packet-slot-get 'number-of-quads 'header))
		  (baq_block_length (* 8 (+ 1 ,(space-packet-slot-get 'baq-block-length 'header))))
		  (number_of_words (static_cast<int> (round (ceil (/ (static_cast<double> (* 10d0 number_of_quads))
								     16)))))
		  (baq_mode ,(space-packet-slot-get 'baq-mode 'header))
		  (fref 37.53472224d0)
		  (swst (/ (static_cast<double> ,(space-packet-slot-get 'sampling-window-start-time 'header))
			   fref))
		  (delta_t_suppressed (/ 320d0 (* 8 fref)))
		  (data_delay_us (+ swst delta_t_suppressed))
		  (data_delay (+ ,(/ 320 8)
				 ,(space-packet-slot-get 'sampling-window-start-time 'header)))
		  (data (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
		  )
	      
	      #+safety (do0
			#+nil (assert (or (== -1 mi_data_delay)
				    (<= mi_data_delay data_delay)))
			;(assert (<= number_of_baq_blocks 256))
			(assert (or ,@(loop for e in `(0) collect
					   `(== ,e baq_mode))))
			;#+log-brc
			,(logprint "" `(packet_idx baq_mode baq_block_length
						   data_delay_us
						   data_delay
						   number_of_quads
						   )))
	      (let ((s))
		(declare (type sequential_bit_t s))
		(init_sequential_bit_function &s (+ (aref ,(g `_header_offset) packet_idx)
						    62 6))
		(let ((data_start s.data))
		 ,@(loop for e in `(ie io qe qo)
		      collect
			(let ((sym
			       (format nil "decoded_~a_symbols" e))
			      (sym-a (format nil "decoded_~a_symbols_a" e)))
			  `(let ((,sym 0)
				 (,sym-a))
			     (declare (type "std::array<float,MAX_NUMBER_QUADS>" ,sym-a))
			     (do0
			      (dotimes (i MAX_NUMBER_QUADS)
				(setf (aref ,sym-a i) 0s0)))
			     (do0
			      ,(format nil "// parse ~a data" e)
			      (dotimes (i number_of_quads)
				(let ((smcode (get_data_type_a_or_b &s))
				      (sign_bit (logand 1 (>> smcode 9))) ;; FIXME
				      (mcode (logand smcode (hex ,(loop for i below 9 sum
								       (expt 2 i)))
						     #+nil (hex #b1 1111 1111)))
				      (scode (* (powf -1s0 (static_cast<float> sign_bit))
							     (static_cast<float> mcode))))
				  (declare (type int sign_bit)
					   (type float scode))
				  (setf (aref ,sym-a ,sym) scode)
				  (incf ,sym)))
			      (consume_padding_bits &s)

			      #+nil (progn
				(let ((word_end (+ 62 2 (* 2 (- number_of_words 1) ,(case e
										  (ie 1)
										  (io 2)
										  (qe 3)
										  (qo 4))))))
				 (setf s.data (+ data_start word_end))))
			      #+nil
			      (progn
				(let ((word_end (+ 62 2 (* 2 (- number_of_words 1) ,(case e
										  (ie 1)
										  (io 2)
										  (qe 3)
										  (qo 4)))))
				      (seq_off (- s.data data_start)))
				  ,(logprint "padding" `(word_end
							 seq_off
							 s.current_bit_count)))))))))
		(do0
		 (assert (== decoded_ie_symbols
			     decoded_io_symbols 
			     ))
		 (assert (== decoded_ie_symbols
			     decoded_qe_symbols
			     ))
		 (assert (== decoded_qo_symbols
			     decoded_qe_symbols
			     )))
		(dotimes (i decoded_ie_symbols)
		  (do0 (dot (aref output (* 2 i)) (real (aref decoded_ie_symbols_a i)))
		       (dot (aref output (* 2 i)) (imag (aref decoded_qe_symbols_a i))))
		  (do0 (dot (aref output (+ 1 (* 2 i))) (real (aref decoded_io_symbols_a i)))
		       (dot (aref output (+ 1 (* 2 i))) (imag (aref decoded_qo_symbols_a i)))))
		(let ((n (+ decoded_ie_symbols
			    decoded_io_symbols)))
		  (return n))))))))


     (let ((l `(,@(loop for e in `(x y z) collect
		    `(,(format nil "~a_axis_position" e) double))
	       ,@(loop for e in `(x y z) collect
		      `(,(format nil "~a_velocity" e) float))
	       ,@(loop for i below 4 collect
		      `(,(format nil "pod_solution_data_stamp_~a" i) uint16_t))
	       ,@(loop for i below 4 collect
		      `(,(format nil "quaternion_~a" i) float))
	       ,@(loop for e in `(x y z) collect
		      `(,(format nil "angular_rate_~a" e) float))
	       ,@(loop for i below 4 collect
		      `(,(format nil "gps_data_timestamp_~a" i) uint16_t))
	       (pointing_status uint16_t)
	       (temperature_update_status uint16_t)
	       ,@(loop for tile from 1 upto 14 by 2 appending
		      `((,(format nil "tile_~a_efe_h_temperature" tile) uint8_t)
			(,(format nil "tile_~a_efe_v_temperature" tile) uint8_t)
			(,(format nil "tile_~a_active_ta_temperature" tile) uint8_t)
			(,(format nil "tile_~a_efe_h_ta_temperature" (+ 1 tile)) uint8_t)
			
			(,(format nil "tile_~a_efe_h_temperature" (+ 1 tile)) uint8_t)
			(,(format nil "tile_~a_efe_v_temperature" (+ 1 tile)) uint8_t)
			(,(format nil "tile_~a_active_ta_temperature" (+ 1 tile)) uint8_t)
			(,(format nil "tile_~a_efe_h_ta_temperature" (+ 2 tile)) uint8_t)))
	       (tgu_temperature uint16_t))))
    (define-module
	`(decode_sub_commutated_data
	  ((_ancillary_data :direction 'out :type "std::array<uint16_t,65>")
	   (_ancillary_data_valid :direction 'out :type "std::array<bool,65>")
	   (_ancillary_decoded :direction 'out :type "ancillary_data_t")
	   (_ancillary_data_index :direction 'out :type int))
	  (do0
	   (include <cstring>
		    <cassert>
		    <fstream>)
	   ,(emit-utils :code
			`(do0
			  (defstruct0 ancillary_data_t
			      ,@l)))
	   (defun init_sub_commutated_data_decoder ()
	     (setf ,(g `_ancillary_data_index) 0)
	     (dotimes (i (static_cast<int> (dot ,(g `_ancillary_data_valid)
			       (size))))
	       (setf (dot ,(g `_ancillary_data_valid)
			  (at i))
		     false)))
	   (defun feed_sub_commutated_data_decoder (word idx space_packet_count) 
	     (declare (type uint16_t word)
		      (type int idx space_packet_count)
		      (values bool)) ;; data full
	     #+nil ,(logprint "add" `(word idx))
	     (setf  ,(g `_ancillary_data_index) idx
		    (dot ,(g `_ancillary_data)
			 (at ,(g `_ancillary_data_index)))
		    word
		    (dot ,(g `_ancillary_data_valid)
			 (at ,(g `_ancillary_data_index)))
		    true)
					;(incf ,(g `_ancillary_data_index))
	     (if (== ,(g `_ancillary_data_index)
		     (- (dot ,(g `_ancillary_data)
			     (size))
			1))
		 (do0
		  ,@(loop for i from 1 upto 64 collect
			 `(unless (dot ,(g `_ancillary_data_valid)
				       (at ,i))
			    (return false)))
		 
		  (memcpy (reinterpret_cast<void*> (ref ,(g `_ancillary_decoded)))
			  (reinterpret_cast<void*> (dot ,(g `_ancillary_data)
							(data)))
			  (sizeof ,(g `_ancillary_data)))
		  (init_sub_commutated_data_decoder)

		  (let (,@(loop for x in l collect
			       (destructuring-bind (name type) x
				 `(,name
				   ,(case type
				     (uint8_t `(static_cast<int> (dot ,(g `_ancillary_decoded)
								      ,name)))
				     (t `(dot ,(g `_ancillary_decoded)
					      ,name)))))))
		   ,(csvprint "./o_anxillary.csv"
			      `(space_packet_count
				,@(loop for x in l collect
				       (destructuring-bind (name type) x
					 name)))))
		  #+nil
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
		      "std::endl"
		      ,@(loop for x in l appending
			     (destructuring-bind (name type) x
			       `((string ,(format nil "    ~a=" name))
				 ,(case type
				    (uint8_t `(static_cast<int> (dot ,(g `_ancillary_decoded)
								     ,name)))
				    (t `(dot ,(g `_ancillary_decoded)
					     ,name)))
				 "std::endl"))))
		  (return true))
		 (do0
		  (return false))))))))
    

     (define-module
     `(decode_type_c_packet
       ()
       (do0
	(include <cassert>
		 <cmath>
		 <thread>)
	(include "vis_04_decode_packet.hpp")
	
	  	  
	;; i need tables for: A{3,4,5}[thidx], NRL{3,4,5}[mcode]
	;; SF[thidx] (already in decode_packet module)

	"extern const std::array<const float, 256> table_sf;"
	
	(do0
	 "// table 5.2-1 simple reconstruction parameter values A"
	 ,@(loop for l in `((3 3 3.12 3.55)
			    (7 7 7 7.17 7.4 7.76)
			    (15 15 15  15 15 15  15.44 15.56  16.11 16.38 16.65))
		  
	      and a in '(3 4 5) collect
		(let ((table (format nil "table_a~a" a)))
		  `(let ((,table (curly ,@l)))
		     (declare (type ,(format nil "const std::array<const float,~a>" (length l)) ,table))))))

	(do0
	 "// table 5.2-2 normalized reconstruction levels"
	 ,@(loop for l in `((.2490 .7681 1.3655 2.1864)
			    (.129 .39 .6601 .9471 1.2623 1.6261 2.0793 2.7467)
			    (.066 .1985 .332 .4677 .6061 .7487 .8964 1.0510 1.2143
				  1.3896 1.58 1.7914 2.0329 2.3234 2.6971 3.2692))
	      and a in '(3 4 5) collect
		(let ((table (format nil "table_nrla~a" a)))
		  `(let ((,table (curly ,@ (mapcar (lambda (x) (coerce x 'single-float)) l))))
		     (declare (type ,(format nil "const std::array<const float,~a>" (length l)) ,table))))))

	,@(loop for a in '(3 4 5) collect
	       `(defun ,(format nil "get_baq~a_code" a) (s)
		  (declare (type sequential_bit_t* s)
			   (values "inline int"))
		  (return (+ ,@(loop for j below a collect
				    `(* (hex ,(expt 2 (- (- a 1) j)))
					(get_sequential_bit s)))))
		  ))
	  
	  
	  ,@(loop for a in `(3 4 5) collect
		 `(defun ,(format nil "init_decode_type_c_packet_baq~a" a)
		      (packet_idx ; mi_data_delay
							    output
							    )
	      (declare (type int packet_idx ;mi_data_delay
			     )
		       #+nil (type "std::array<std::complex<float>,MAX_NUMBER_QUADS>&" output)
		       (type "std::complex<float>*" output)
		       (values int))
	    
	      (let ((header (dot (aref ,(g `_header_data) packet_idx)
				 (data)))
		    (offset (aref ,(g `_header_offset) packet_idx))
		    (number_of_quads ,(space-packet-slot-get 'number-of-quads 'header))
		    (baq_block_length (* 8 (+ 1 ,(space-packet-slot-get 'baq-block-length 'header))))
		  
		    (number_of_baq_blocks (static_cast<int> (round (ceil (/ (* 2d0 (static_cast<double> number_of_quads))
									    256)))))
		    
		    (thidxs)
		    (baq_mode ,(space-packet-slot-get 'baq-mode 'header))
		    (fref 37.53472224d0)
		    (swst (/ (static_cast<double> ,(space-packet-slot-get 'sampling-window-start-time 'header))
			     fref))
		    (delta_t_suppressed (/ 320d0 (* 8 fref)))
		    (data_delay_us (+ swst delta_t_suppressed))
		    (data_delay (+ ,(/ 320 8)
				   ,(space-packet-slot-get 'sampling-window-start-time 'header)))
		    #+nil (data_offset (- data_delay mi_data_delay))
			   
		    (data (+ offset (static_cast<uint8_t*> ,(g `_mmap_data))))
		    )
		(declare 
			 (type ,(format nil "std::array<uint8_t,~d>"
					(round (/ max-number-quads 256)))
			       thidxs))

		#+nil (when (== -1 mi_data_delay)
			(setf data_offset 0))
		#+safety (do0
			  #+nil (assert (or (== -1 mi_data_delay)
					    (<= mi_data_delay data_delay)))
			  (assert (<= number_of_baq_blocks 256))
			  (assert (or ,@(loop for e in `(0 3 4 5 12 13 14) collect
					     `(== ,e baq_mode))))
			  #+log-brc ,(logprint "" `(packet_idx baq_mode baq_block_length
							       data_delay_us
							       data_delay
							       number_of_quads
							       )))
		(let ((s))
		  (declare (type sequential_bit_t s))
		  (init_sequential_bit_function &s (+ (aref ,(g `_header_offset) packet_idx)
						      62 6))
		  ,@(loop for e in `(ie io qe qo)
		       collect
			 (let ((sym
				(format nil "decoded_~a_symbols" e))
			       (sym-a (format nil "decoded_~a_symbols_a" e)))
			   `(let ((,sym 0)
				  (,sym-a))
			      (declare (type "std::array<float,MAX_NUMBER_QUADS>" ,sym-a))
			      (do0
			       (dotimes (i MAX_NUMBER_QUADS)
				 (setf (aref ,sym-a i) 0s0)))
			      (do0
			       ,(format nil "// parse ~a data" e)
			       (for ((= "int block" 0)
				     (< ,sym number_of_quads)
				     (incf block))
				    ,(case e
				       (ie
					"// nothing for ie")
				       (io
					"// nothing for io")
				       (qe
					`(let ((thidx (get_threshold_index &s)))
					   (setf (aref thidxs block) thidx)))
				       (qo
					`(let ((thidx (aref thidxs block))))))
				    (progn
				      #+safety (do0
						#+nil (unless (or ,@(loop for e in `(0 1 2 3 4) collect
									 `(== ,e brc)))
							,(logprint "error: out of range" `(brc)) 
							(assert 0))
						
						#+log-brc ,(logprint (format nil "~a" e) `((static_cast<int> brc) block number_of_baq_blocks ,(if (member e `(qe qo))
																		  `(static_cast<int> thidx)
																		  1)))
						)
				      ,(let ((th (case a
						   (3 3)
						   (4 5)
						   (5 10))))
					 `(,@(if (member e `(ie io))
						 `(do0)
						 `(if (<= thidx ,th)))
					     ,@(loop for thidx-choice in (if (member e `(ie io))
									     `(thidx-unknown)
									     `(simple normal)) collect
						    `(do0
						      ,(format nil "// reconstruction law block=~a thidx-choice=~a" e
							       thidx-choice)
						      (for ((= "int i" 0)
							    (and (< i
								    128 ;(/ baq_block_length 2) ;; divide by two because even and odd samples are handled in different loops?
								    )
								 (< ,sym
								    number_of_quads
								    ))
							    (incf i))
							   (let (
								 (smcode (,(format nil "get_baq~a_code" a) &s))
								 (sign_bit (logand 1 (>> smcode (- ,a 1))))
								 (mcode (logand smcode (hex ,(loop for i below (- a 1) sum ;; FIXME: bounds
												  (expt 2 i)))))
								 (mcode_f (static_cast<float> mcode))
								 (symbol_sign 1s0))
							     
							     (when sign_bit
							       (setf symbol_sign -1s0))
							     (do0
							      ,(if (member e `(qe qo))
								   `(do0
								     ,(format nil "// decode ~a p.66" e)
								     ,(let ((th-mcode (case a
											(3 3)
											(4 7)
											(5 15))))
									`(let ((v 0s0))
									   ,(case thidx-choice
									      (simple
									       `(handler-case
										    (do0
										     (if (< mcode ,th-mcode)
											 (setf v (* symbol_sign mcode_f))
											 (if (== mcode ,th-mcode)
											     (setf v (* symbol_sign
													(dot
													 ,(format nil "table_a~a"
														  a)
													 (at thidx))))
											     (do0
											      ,(logprint "mcode too large" `(mcode))
											      (assert 0)))))
										  ("std::out_of_range&" (e)
										    ,(logprint
										      (format nil "exception simple a=~a"
											      a)
										      `(thidx packet_idx))
										    (assert 0))))
									      (normal
									       `(handler-case
										    (do0 (setf v (* symbol_sign
												    (dot ,(format nil
														  "table_nrla~a"
														  a)
													 (at mcode))
												    (dot table_sf (at thidx)))))
										  ("std::out_of_range&" (e)
										    ,(logprint
										      (format nil "exception normal nrl or sf "
											      )
										      `(thidx packet_idx))
										    (assert 0))))))))
								   `(let ((v (* symbol_sign mcode_f)))
								      "// in ie and io we don't have thidx yet, will be processed later"))
							      
							      (setf (aref ,sym-a ,sym)
								    v)))
							   (incf ,sym))))))))
			       (consume_padding_bits &s)))))
		  ,(logprint "decode ie and io blocks" `(number_of_baq_blocks))
		  ,@(loop for e in `(ie io) collect
			 `(dotimes (block number_of_baq_blocks)
			    (let ((thidx (aref thidxs block)))
			      (do0
			       ,(let ( ;(sym (format nil "decoded_~a_symbols" e))
				      (sym-a (format nil "decoded_~a_symbols_a" e)))
				  `(do0
				    #+safety (do0
						  #+nil (unless (or ,@(loop for e in `(0 1 2 3 4) collect
									   `(== ,e brc)))
							  ,(logprint "error: out of range" `(brc)) 
							  (assert 0))
						  #+nil ,(logprint (format nil "~a" e) `((static_cast<int> brc) block number_of_baq_blocks))
						  )
					,(format nil "// decode ~a p.66 reconstruction law middle choice a=~a" e
						 a)
					,(let ((th (case a
						   (3 3)
						   (4 5)
						   (5 10))))
					   `(if (<= thidx ,th)
						,@(loop for thidx-choice in `(simple normal) collect
						       `(do0
							 ,(format nil "// decode ~a p.66 reconstruction law ~a a=~a"
								  e thidx-choice a)
							 (for ((= "int i" 0)
							       (and (< i 128)
								    (< (+ i (* 128 block))
								       ,(format nil "decoded_~a_symbols" e)))
							       (incf i)) ;dotimes (i 128)
							      (let ((pos
								     (+ i (* 128 block)))
								    (scode (aref ,sym-a pos))
								    (mcode_f (fabsf scode))
								    (mcode (static_cast<int> mcode_f))
								    (symbol_sign (copysignf 1s0 scode)))
								(do0
								 ,(format nil "// decode ~a p.66 reconstruction law right side" e)
								 ,(let ((th-mcode (case a
											(3 3)
											(4 7)
											(5 15))))
								    `(let ((v 0s0))
								       ,(case thidx-choice
									  (simple
									   `
									   (handler-case
									       (do0 (if (< mcode ,th-mcode)
											(setf v (* symbol_sign mcode_f))
											(if (== mcode ,th-mcode)
											    (setf v (* symbol_sign
												       (dot
													,(format nil "table_a~a"
														 a)
													(at thidx))))
											    (do0
											     #-nolog ,(logprint "mcode too large" `(mcode))
											     (assert 0)))))
									     ("std::out_of_range&" (e)
									       ,(logprint
										 (format nil "exception simple block=~a a=~a"
											 e a)
										 `((static_cast<int> thidx) mcode packet_idx))
									       (assert 0))))
									  (normal
									   `
									   (handler-case
									       (do0 (setf v (* symbol_sign
											       (dot ,(format nil
													     "table_nrla~a"
													     a)
												    (at mcode))
											       (dot table_sf (at thidx)))))
									     ("std::out_of_range&" (e)
									       ,(logprint
										 (format nil "exception normal nrl or sf block=~a a=~a"
											 e a)
										 `((static_cast<int> thidx) block i
										   mcode packet_idx pos scode symbol_sign
										   
										   ,(format nil "decoded_~a_symbols" e)
										   ))
									       (assert 0)))))))
								 (setf (aref ,sym-a pos) v))))))))))))))
		  (do0
		 (assert (== decoded_ie_symbols
			     decoded_io_symbols
			     ))
		 (assert (== decoded_ie_symbols
			     decoded_qe_symbols
			     ))
		 (assert (== decoded_qo_symbols
			     decoded_qe_symbols
			     )))
		  (dotimes (i decoded_ie_symbols)
		    (do0 (dot (aref output (* 2 i)) (real (aref decoded_ie_symbols_a i)))
			 (dot (aref output (* 2 i)) (imag (aref decoded_qe_symbols_a i))))
		    (do0 (dot (aref output (+ 1 (* 2 i))) (real (aref decoded_io_symbols_a i)))
			 (dot (aref output (+ 1 (* 2 i))) (imag (aref decoded_qo_symbols_a i)))))
		  (let ((n (+ decoded_ie_symbols
			      decoded_io_symbols)))
		    (return n)))))))))
     
  )
  
  (progn
    (progn
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 (let* ((file (format nil
				      "vis_~2,'0d_~a"
				      i name
				      ))
			(file-h (string-upcase (format nil "~a_H" file))))
		   (with-open-file (sh (asdf:system-relative-pathname 'cl-cpp-generator2
								      (format nil "~a/~a.hpp"
									      *source-dir* file))
				       :direction :output
				       :if-exists :supersede
				       :if-does-not-exist :create)
		     (format sh "#ifndef ~a~%" file-h)
		     (format sh "#define ~a~%" file-h)
		     
		     (emit-c :code code
			     :hook-defun #'(lambda (str)
					     (format sh "~a~%" str)
					     )
			     :hook-defclass #'(lambda (str)
						(format sh "~a;~%" str)
						)
			     :header-only t
			     )
		     (format sh "#endif"))))

	       #+nil (format t "emit cpp file for ~a~%" name)
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/vis_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code))))
      #+nil (format s "#endif"))
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
			  e))
		    " "
		    "#endif"
		    " "))
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "

		    #+nil (include <complex>)
		    #+nil (include <deque>
			     <map>
			     <string>)
		    #+nil (include <thread>
			     <mutex>
			     <queue>
			     <condition_variable>
			     )
		    " "

		    " "
		    ;(include "proto2.h")
		    " "
		    ,@(loop for e in (reverse *global-code*) collect
			 e)

		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))



