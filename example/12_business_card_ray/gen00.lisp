(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(setf *features* (union *features* '()))
(setf *features* (set-difference *features* '()))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/12_business_card_ray/source/")

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
	     #+nil (- (dot ("std::chrono::high_resolution_clock::now")
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
      (let ((l `(#+nil (_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
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
      `(cuda_main (
	      )
		  (do0
		   "// /opt/cuda/bin/nvcc nvcut_00_cuda_main.cu  -I/opt/cuda/include/ --std=c++14 -O1 -g -Xcompiler=-march=native --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0" 
		   
	      
	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))

	      "enum { DIM=512, BPP=3};"

	      "using namespace std::chrono_literals;"
	      ; "using namespace filament;"


	      (space "struct v"
		     (progn
		       (let ((x)
			     (y)
			     (z))
			 (declare (type float x y z)))
		       (space __device__
			(defun operator+ (r)
			  (declare (type v r)
				   (values v))
			  (return (v ,@(loop for e in `(x y z) collect
					    `(+ ,e (dot r ,e)))))))
		       (space __device__
			(defun operator* (r)
			  (declare (type float r)
				   (values v))
			  (return (v ,@(loop for e in `(x y z) collect
					    `(* ,e r))))))
		       (space __device__
			(defun operator% (r)
			  (declare (type v r)
				   (values float))
			  (return (+ ,@(loop for e in `(x y z) collect
					    `(* ,e (dot r ,e)))))))
		       (space __device__
			(defun operator^ (r)
			  (declare (type v r)
				   (values v))
			  (return (v ,@(loop for (e f) in `((y z) (z x) (x y)) collect
					    `(- (* ,e (dot r ,f))
						(* ,f (dot r ,e))))))))
		       (space __device__
			(defun v ()
			  (declare 
			   (values " "))))

		       (space __device__
			(defun v (a b c)
			  (declare (type float a b c)
				   (values " "))
			  ,@(loop for e in `(x y z) and f in `(a b c) collect
					    `(setf ,e ,f))
			  ))

		       (space __device__
			(defun "operator!" ()
			  (declare 
				   (values v))
			  (return (* *this
				     (/ 1s0 (sqrt (% *this *this)))))))

		       ))


	      (do0
	       ,(let ((l `(247570 280596 280600 249748 18578 18577 231184 16 16)))
		  `(let ((G (curly ,@l))
			 (g_seed 1))
		     (declare (type (array  "__device__ int" ,(length l))
				    G)
			      (type "__device__ int" g_seed))
		     (space __device__
			    (defun R ()
			      (declare (values float))
			      (setf g_seed (+ (* 214013 g_seed)
					      2531011))
			      (return (/ (& (>> g_seed 16)
					    0x7fff)
					 66635s0)))
			    ))))

	      (space __device__
		     (defun Sample (origin destination r)
		       (declare (values v)
				(type v origin destination)
				(type int r))
		       (let ((color 1s0))
			(return (v color color color)))))
	      
	      (space __global__
		     (defun GetColor (img)
		       (declare (values void)
				(type "unsigned char*" img))
		       (let ((x blockIdx.x)
			     (y threadIdx.x)
			     (cam_dir (! (v -6s0 -16s0 0s0)))
			     (s .002s0)
			     (cam_up (* (! (^ (v 0s0 0s0 1s0) cam_dir)) s))
			     (cam_right (* (! (^ cam_dir cam_up)) s))
			     (eye_offset (+  (* (+ cam_up
						cam_right)
					       -256)
					    cam_dir))
			     (color (v 13s0 13s0 13s0)))
			 (dotimes (r 64)
			   (let ((delta (+ (* cam_up
					      (- (R) .5s0)
					      99)
					   (* cam_right
					      (- (R) .5s0)
					      99))))
			     #+nil
			     (incf color
				   ))))))
	      	      
	      (defun main ()
		(declare (values int))
		(let ((bitmap (aref "new char"
				    (* DIM DIM BPP)))
		      (dev_bitmap))
		  (declare (type "unsigned char*" dev_bitmap)
			   (type char* bitmap))
		 (do0
		  (cudaMalloc (reinterpret_cast<void**> &dev_bitmap)
				  (* DIM DIM BPP))
		      ("GetColor<<<DIM,DIM>>>" dev_bitmap)
		      (cudaMemcpy bitmap dev_bitmap (* DIM DIM BPP)
				  cudaMemcpyDeviceToHost)
		      (delete bitmap)))
		(return 0 ; EXIT_SUCCESS
			)))))

  
  
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
				      "~a/nvcut_~2,'0d_~a.~a"
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
		    (include <iostream>
			     ;<iomanip>
			     )
		    		    
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

		    " "

		    
		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))

