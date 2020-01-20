(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(setf *features* (union *features* '()))
(setf *features* (set-difference *features* '()))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/09_concurrent_producer_fsm/source/")

  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun logprint (msg &optional rest)
      `(do0
	#-nolog
	(do0
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
	     "std::endl"))))
    
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  `(,name ,type))))))
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
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))


    
  (define-module
      `(main ((_q :type "FixedDeque<10>"))
	     (do0
	      
	      (include <cstdlib>
		       <experimental/random>)
	      " "

	      #+nil
	      (do0

	       (_serial_communication_queue_incomment_mutex :type "std::mutex")
	       (_serial_communication_queue_incomment_filled_condition :type "std::condition_variable")
	       

	       ,(format nil
			"std::lock_guard<std::mutex> guard(~a);"
			(cl-cpp-generator2::emit-c :code (g `_virtual_twin_data_mutex)))
	       
	       ,(format nil
			"std::unique_lock<std::mutex> lk(~a);"
			(cl-cpp-generator2::emit-c :code (g `_serial_communication_queue_out_mutex)))

	       (dot ,(g `_serial_communication_queue_out_filled_condition)
		    (wait lk)))

	      

	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))


	      "using namespace std::chrono_literals;"
	      	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))
		
		(let ((th0 ("std::thread"
			   (lambda ()
			     (declare (values float))
			     ("std::this_thread::sleep_for" "30ms")
			     (<< "std::cout"
				 (string "hello ")
				 "std::endl")
			     (dotimes (i 10)
			       (<< "std::cout"
				   (string "push ")
				   i
				   "std::endl"
				   "std::flush")
			       ("std::this_thread::sleep_for" (* ("std::experimental::fundamentals_v2::randint" 5 300) "1ms"))
			       (dot ,(g `_q)
				    (push_back (* 1s0 i)))
			       )
			     (return 2s0))))
		      (th1 ("std::thread"
			   (lambda ()
			     (declare (values float))
			     
			     (dotimes (i 12)
			       (let ((b (dot ,(g `_q)
					 (back))))
				(<< "std::cout"
				    (string "                  back=")
				    b
				    (string " ["))
				(progn
				  ,(format nil
				  "std::lock_guard<std::mutex> guard(~a);"
				  (cl-cpp-generator2::emit-c :code `(dot ,(g `_q)
									 mutex)))
				  (dotimes (i (dot ,(g `_q)
						   (size)))
				   (<< "std::cout"
				       (aref ,(g `_q) i)
				       (string " "))))
				(<< "std::cout"
				    (string "]")
				    "std::endl"
				    "std::flush"))
			       )
			     (return 2s0))))))
		(th0.join)
		(th1.join)
		(return 0)))))

  
  
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						 (merge-pathnames #P"proto2.h"
								  *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       (unless cuda
		(emit-c :code code :hook-defun 
			#'(lambda (str)
			    (format s "~a~%" str))))
	       
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/fsm_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code)))))
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
		    " "
		    
		    )
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

		    (include <deque>
		       <thread>
		       <condition_variable>
		       )
		    " "

		    (do0
		     "template <int MaxLen>"
		     (defclass FixedDeque "public std::deque<float>"
		       "public:" ; "private:"
		       (let ((filled_condition)
			     (mutex))
			 (declare (type "std::condition_variable" filled_condition)
				  (type "std::mutex" mutex)))
		       "public:"
		       (defun push_back (val)
			 (declare (type "const float&" val))
			 ,(format nil
				  "std::lock_guard<std::mutex> guard(~a);"
				  (cl-cpp-generator2::emit-c :code 'this->mutex))
			 (when (== MaxLen (this->size))
			   (this->pop_front))

			 ("std::deque<float>::push_back" val)
			 (dot this->filled_condition
					   (notify_all)))
		       (defun back ()
			 (declare (values float))
			 ,(format nil
				  "std::unique_lock<std::mutex> lk(~a);"
				  (cl-cpp-generator2::emit-c :code 'this->mutex))
			 (dot this->filled_condition
			      (wait lk))
			 (let ((b ("std::deque<float>::back")))
			   (lk.unlock)
			   (return b)))))
		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))

