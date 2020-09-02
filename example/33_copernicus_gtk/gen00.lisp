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

  (let*  ()
    
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
			     
			     )

		    ;(include <gtkmm.h>)
		    " "

		    
		    
		    (split-header-and-code
		     (do0
		      "// header"
		      
		      (include <gtkmm/treeview.h>
			       <gtkmm/liststore.h>
			       <gtkmm/box.h>
			       <gtkmm/scrolledwindow.h>
			       <gtkmm/window.h>
			       )
		      #+nil (include <gtkmm.h>)
		      " "
		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp"
			       "vis_01_mmap.hpp")
		      
		      " "
		      ))

		    (let ((state ,(emit-globals :init t)))
		      (declare (type "State" state)))
		    
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
		      
		      )

		    

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
		      
		      (let ((app (Gtk--Application--create argc argv
							   (string "org.gtkmm.example")))
			    (hw))
			(declare (type Example_TreeView_ListStore ;HelloWorld
				       hw))
			;(win.set_default_size 200 200)
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



