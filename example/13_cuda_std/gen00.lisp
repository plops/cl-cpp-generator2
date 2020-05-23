(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(setf *features* (union *features* '()))
(setf *features* (set-difference *features* '()))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/13_cuda_std/source/")
  (defparameter *repo-sub-path* "13_cuda_std")
  (defparameter *inspection-facts*
    `((10 "")))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

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
		   "// /opt/cuda/bin/nvcc custd_00_cuda_main.cu --gpu-architecture=compute_75 --gpu-code=compute_75 --use_fast_math  -I/opt/cuda/include/ --std=c++14 -O3 -g -Xcompiler=-march=native --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0"
		   "// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/cwe21285.pdf p. 338"

		   (include <cstdio>)
	      (let (
		    (_code_git_version
		     (string ,(let ((str 
				     #-sbcl "xxx"
				     #+sbcl (with-output-to-string (s)
					      (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				(subseq str 0 (1- (length str))))))
		    (_code_repository (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/example/~a/source/" *repo-sub-path*)
					      ))

		    (_code_generation_time
		     (string ,(multiple-value-bind
				    (second minute hour date month year day-of-week dst-p tz)
				  (get-decoded-time)
				(declare (ignorable dst-p))
				(format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
					hour
					minute
					second
					(nth day-of-week *day-names*)
					year
					month
					date
					(- tz)))))))

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
				     (/ 1s0 (sqrtf (% *this *this)))))))

		       ))


	      (do0
	       (let (
		      (g_seed 1))
		     (declare 
			      (type "__device__ int" g_seed))
		     (space __device__
			    (defun R ()
			      (declare (values float))
			      (setf g_seed (+ (* 214013 g_seed)
					      2531011))
			      (return (/ (& (>> g_seed 16)
					    0x7fff)
					 66635s0)))
			    )))

	      (do0
	       ,(let ((l `(247570 280596 280600 249748 18578 18577 231184 16 16)))
		  `(let ((G (curly ,@l))
			 )
		     (declare (type (array  "__device__ int" ,(length l))
				    G)
			      )
		     (space __device__
			    (defun TraceRay (origin destination tau normal)
			      (declare (values int)
				       (type v origin destination)
				       (type float& tau)
				       (type v& normal)
				       )

			      (setf tau 1s9)
			      (let ((m 0)
				    (p (/ -origin.z
					  destination.z)))
				(when (< .01s0 p)
				  (setf tau p
					normal (v 0 0 1)
					m 1))
				(for ((= "int k" 19)
				      (< 0 k)
				      "k--")
				     (for ((= "int j" 9)
					   (< 0 j)
					   "j--")
					  (when #+nil (<<
						 (& (aref G j) 1)
						 k
						 )
						(& (aref G j) (<< 1 k))
						
						
					    (let ((p (+ origin (v -k 0 (- -j 4))))
						  (b (% p destination))
						  (c (- (% p p) 1s0))
						  (q (- (* b b) c)))
					      (when (< 0 q)
						;; ray hits sphere
						(let ((s (- -b (sqrtf q))))
						  ;; distance camera-sphere
						  (when (and (< s tau)
							    (< .01s0 s))
						    (setf tau s
							  normal (! (+ p (* destination tau)))
							  m 2))))))))
				(return m)))))))

	      (space __device__
		     (defun Sample (origin destination r)
		       (declare (values v)
				(type v origin destination)
				(type int r))

		       (let ((tau 0s0)
			     (normal (v)))
			 (when (< 4 r)
			   (return (v 0 80 0)))
			 (let ((match (TraceRay origin
						destination
						tau normal)))
			   (unless match
			     ;; no sphere hit, ray goes up
			     (return (v 0 0 80)#+nil (* (v .7s0 .6s0 1)
						(powf (- 1 destination.z) 4))))
			   ;; a sphere maybe hit
			   (let ((intersection (+ origin (* destination tau)))
				 (light_dir (+ (! (v (+ 9 (R))
						     (+ 9 (R))
						     16))
					       (* intersection -1)))
				 (half_vec (+ destination
					      (* normal (* (% normal
							      destination)
							   -2))))
				 (lamb_f (% light_dir normal)))
			     ;; lambertian coef > 0 or in shadow
			     (when (or (< lamb_f 0)
				       (TraceRay intersection light_dir tau normal))
			       (setf lamb_f 0))


			     (when (& match 1)
				 ;; no sphere hit and ray goes down
			       (setf intersection (* intersection .2s0))
			       (let ((c (? (& (static_cast<int> (+ (ceilf intersection.x)
								   (ceilf intersection.y)))
					      1)
					   (v 3 1 1)
					   (v 3 3 3)
					      )))
				(return (* c (+ (* lamb_f .2s0)
						 .1s0)))))
			     
			     (let ((color (powf #+nil (* (% light_dir
						      (* half_vec
							 (< 0 lamb_f)))
							 1)
						(* (% light_dir
						    half_vec
						    )
						   1 ;(< lamb_f 0)
						   )
						
						
						99s0)))
			       (declare (type float color))
			       
			       ;; m == 2 sphere was hit, cast ray bouncing from sphere, attenuate color by 50%
			       (return ; (v color color color)
				       #-nil (+ (v color color color)
						(* (Sample intersection half_vec (+ r 1))
						   .5s0)))
			       ))
			   ))
		       ))
	      
	      (space __global__
		     (defun GetColor (img)
		       (declare (values void)
				(type "unsigned char*" img))
		       (let ((x blockIdx.x)
			     (y threadIdx.x)
			     (cam_dir (! (v -2s0 -16s0 0s0)))
			     (s .004s0)
			     (cam_up (* (! (^ (v 0s0 0s0 1s0) cam_dir)) s))
			     (cam_right (* (! (^ cam_dir cam_up)) s))
			     (eye_offset (+  (* (+ cam_up
						cam_right)
					       -256)
					    cam_dir))
			     (color (v 13s0 13s0 13s0)))
			 (dotimes (r 64)
			   (let ((delta (+ (* cam_up
					      ;(- (R) .5s0)
					      99)
					   (* cam_right
					      ;(- (R) .5s0)
					      99))))
			     
			     (setf color
				   (+
				    (* 
				     (Sample (+ (v 17 16 8)
						delta)
					     (! (* (+ (* delta -1)
						      (* cam_up x ;(+ (R) x)
							 )
						      (* cam_right y ; (+ y (R))
							 )
						      eye_offset)
						   16))
					     0)
				     3.5s0)
				    color))))
			 ,@(loop for e in `(x y z) and i from 0 collect
				`(setf (aref img (+ (* DIM y BPP)
						    (* BPP x)
						    ,i))
				       (dot color ,e))))))
	      	      
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
		      (do0
		       (printf (string "P6 512 512 255 "))
		       (let ((c bitmap))
			 (for ((= "int y" DIM)
			       y
			       "y--") ; dotimes (y DIM)
			      (for ((= "int x" DIM)
				    x
				    "x--") ; dotimes (x DIM)
				    (setf c (ref (aref bitmap (+ (* y DIM BPP)
								 (* x BPP)))))
				    (printf (string "%c%c%c")
					    (aref c 0)
					    (aref c 1)
					    (aref c 2))
				    (incf c BPP)))))
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
				      "~a/custd_~2,'0d_~a.~a"
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

