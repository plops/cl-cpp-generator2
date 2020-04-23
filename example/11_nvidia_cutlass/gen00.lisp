(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(setf *features* (union *features* '()))
(setf *features* (set-difference *features* '()))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/11_nvidia_cutlass/source/")

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
		   "/*"
		   "  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/martin/src/cutlass/build/tools/library/"
		   "  export PATH=$PATH:/opt/cuda/nvvm/bin/"
		   "  /opt/cuda/bin/nvcc nvcut_00_cuda_main.cu  -I /home/martin/src/cutlass/include/ -I /opt/cuda/include/ -I/home/martin/src/cutlass/tools/util/include/ -I/home/martin/src/tools/library/include -L/home/martin/src/cutlass/build/tools/library/ -lcutlass --std=c++14 -O1 -g -Xcompiler=-march=native --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0"
		   "*/"
	      "// https://github.com/NVIDIA/cutlass/blob/master/media/docs/quickstart.md"
	      "// https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md"
	      (include <cutlass/numeric_types.h>
		       <cutlass/gemm/device/gemm.h>
		       <cutlass/util/host_tensor.h>)
	      " " 
	      
	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))
	      

	      "using namespace std::chrono_literals;"
	      ; "using namespace filament;"
	      
	      	      
	      (defun main ()
		(declare (values int))
		#+nil (setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
						   (time_since_epoch)
						   (count)))


		(setf "using Gemm"
		      (space
		       cutlass--gemm--device--Gemm
		       (angle cutlass--half_t
			      cutlass--layout--ColumnMajor
			      cutlass--half_t
			      cutlass--layout--ColumnMajor
			      cutlass--half_t
			      cutlass--layout--ColumnMajor
			      float
			      cutlass--arch--OpClassTensorOp
			      cutlass--arch--Sm75
			      )))
		(let ((gemm_op)
		      (status))
		  (declare (type Gemm gemm_op)
			   (type cutlass--Status status)))

		,(let ((l `((A) (B) (C) (D C ""))))
		   `(let ((M 512)
			  (N 256)
			  (K 128)
			  (alpha (float 1.25s0))
			  (beta (float -1.25s0))
			  ((A (curly M K)))
			  ((B (curly K N)))
			  ((C (curly M N)))
					;((D (curly M N)))
			  ,@(loop for e in l
			       collect
				 (destructuring-bind (ptr &optional (var ptr) (const 'const)) e
				   `(,(format nil "*ptr~a" ptr) (,(format nil "static_cast<cutlass::half_t ~a*>" const)
								  (dot ,var (device_data))))))
			  
			  
			  ,@(loop for e in l collect
				 (destructuring-bind (ptr &optional (var ptr) (const 'const)) e
				   (declare (ignorable const))
				  `(,(format nil "ld~a" ptr) (dot ,var (device_ref) (stride 0)))))
			  
			  )
		      (declare (type (space cutlass--HostTensor (angle
								 cutlass--half_t
								 cutlass--layout--ColumnMajor))
				     (A (curly M K))
				     (B (curly K N))
				     (C (curly M N))
				     (D (curly M N))
				     ))
		      (setf status (gemm_op (curly (curly M N K)
						   ,@(loop for e in `((ptrA ldA)
								      (ptrB ldB)
								      (ptrC ldC)
								      (ptrD ldD)
								      (alpha beta))
							collect
							  `(curly ,@e)))))
		      (unless (== status cutlass--Status--kSuccess)
			(return -1))))
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

