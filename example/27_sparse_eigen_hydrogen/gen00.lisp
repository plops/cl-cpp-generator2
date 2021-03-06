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
  (defparameter *source-dir* #P"example/27_sparse_eigen_hydrogen/source/")
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  
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
  (defun cuss (code)
    `(unless (== CUDA_SUCCESS
		 ,code)
       (throw (std--runtime_error (string ,(format nil "~a" (emit-c :code code)))))))
  (defun rtc (code)
    `(unless (== NVRTC_SUCCESS
		 ,code)
       (throw (std--runtime_error (string ,(format nil "~a" (emit-c :code code)))))))
  (defun cuda (code)
    `(progn
       (let ((res ,code))
	(unless (== cudaSuccess
		    res)
	  
	  ,(logprint "error:" `((cudaGetErrorString res)))
	  (throw (std--runtime_error (string ,(format nil "~a" (emit-c :code code)))))))))

  (let*  ((N 3000)
	  (rmax 50s0)
	  (dr (/ rmax N))
	  (cuda-free nil))
   (define-module
       `(cuda_main ((_main_version :type "std::string")
		    (_code_repository :type "std::string")
		    (_code_generation_time :type "std::string")
		    )
		   (do0
		    (include <iostream>
			     <chrono>
			     <cstdio>
			     <cassert>
					;<unordered_map>
			     <string>
			     <fstream>
			     <thread>
			     <vector>
			     <experimental/iterator>
			     <algorithm>
			     )
		    " "
		    (include
					;<nvrtc.h>
		     <cuda.h>
		     <cuda_runtime.h>)
		    " "
					;(include <yacx/main.hpp>)

		    #+nil(space "extern \"C\"" (progn
						 (include "arpack-ng/b/arpackdef.h")))
		    (include "arpack-ng/ICB/arpack.hpp")
		    (include "arpackpp/include/arrssym.h")
	      
		    " "

		    
	      
		    "using namespace std::chrono_literals;"
		    (let ((state ,(emit-globals :init t)))
		      (declare (type "State" state)))

		    
		    (defun kernel_hamiltonian (out in)
		      (declare (type float* out in)
			       (values "__global__ void"))
		      (let ((idx (+ (* blockIdx.x blockDim.x)
				    threadIdx.x))
			    (ri (* (+ 1 idx) ,dr))
			    (l 0)
			    (Z 1)
			    )
			(when (< idx ,N)
			  (let ((Vr (- (/ (* l (+ l 1))
					  (* ri ri))
				       (/ (* 2 Z)
					  ri))))
			    (if (<= 1 idx ,(- N 2))
				(setf (aref out idx)
				      (+ (* (/ -1 ,(* dr dr))
					    (+ (aref in (- idx 1))
					       (aref in (+ idx 1))))
					 (* (+ (/ 2 ,(* dr dr))
					       Vr)
					    (aref in idx))))
				(if (== idx 0)
				    (setf (aref out idx)
					  (+ (* (/ -1 ,(* dr dr))
						(+ ;(aref in (- idx 1))
						 (aref in (+ idx 1))))
					     (* (+ (/ 2 ,(* dr dr))
						   Vr)
						(aref in idx))))
				    (setf (aref out idx)
					  (+ (* (/ -1 ,(* dr dr))
						(+ (aref in (- idx 1))
					;(aref in (+ idx 1))
						   ))
					     (* (+ (/ 2 ,(* dr dr))
						   Vr)
						(aref in idx))))))))))
		    
		    
		    (defun main (argc argv)
		      (declare (values int)
			       (type int argc)
			       (type "char const *const *const" argv))
		      (setf ,(g `_main_version)
			    (string ,(let ((str (with-output-to-string (s)
						  (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				       (subseq str 0 (1- (length str))))))

		      (setf
		       ,(g `_code_repository) (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/example/27_sparse_eigen_hydrogen"))
		       ,(g `_code_generation_time) 
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
					  (- tz)))))

		      (setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
						   (time_since_epoch)
						   (count)))
		      ,(logprint "start main" `(,(g `_main_version)))
		      ,(logprint "" `(,(g `_code_repository)))
		      ,(logprint "" `(,(g `_code_generation_time)))


		      
		      
		      (let ((stream)
			    (blocks ,(ceiling N 512))
			    (threads 512) ;; threads per block
			    )
			(declare (type cudaStream_t stream))
			,(cuda `(cudaStreamCreate &stream))
			
			
			(let ((in)
			      (out))
			  (declare (type float* in out))
			  ,(progn
			     (push 'in cuda-free)
			     (push 'out cuda-free)
			     `(do0 ,(cuda `(cudaMallocManaged &in (* ,N (sizeof float))))
				   ,(cuda `(cudaMallocManaged &out (* ,N (sizeof float))))))


			  (comments "relevant arpack++ example https://github.com/m-reuter/arpackpp/blob/master/examples/reverse/sym/rsymreg.cc")
			  ;; get one eigenvector

			  (comments "The following values of which are available:"
"which = 'LM' : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in the euclidean norm of complex numbers."
"which = 'SM' : Eigenvalues with smallest magnitude (eigs, eigsh), that is, smallest eigenvalues in the euclidean norm of complex numbers."
"which = 'LR' : Eigenvalues with largest real part (eigs)."
"which = 'SR' : Eigenvalues with smallest real part (eigs)."
"which = 'LI' : Eigenvalues with largest imaginary part (eigs)."
"which = 'SI' : Eigenvalues with smallest imaginary part (eigs)."
"which = 'LA' : Eigenvalues with largest algebraic value (eigsh), that is, largest eigenvalues inclusive of any negative sign."
"which = 'SA' : Eigenvalues with smallest algebraic value (eigsh), that is, smallest eigenvalues inclusive of any negative sign."
"which = 'BE' : Eigenvalues from both ends of the spectrum (eigsh)."
"Note that ARPACK is generally better at finding extremal eigenvalues, that is, eigenvalues with large magnitudes. In particular, using which = 'SM' may lead to slow execution time and/or anomalous results. A better approach is to use shift-invert mode.")
			  
			  (let ((prob (ARrcSymStdEig<float> ,N ;; n
							    "4L" ;; nevp
							    (string "SA") ;; which
							    0 ;; ncvp
							    0.0s0  ;; tolp
							    100000 ;; maxitp 
							    )))
			    (while (not (prob.ArnoldiBasisFound))
			      (prob.TakeStep)
			      (let ((ido (prob.GetIdo)))
				(when (or (== ido 1)
					  (== ido -1))
				  (let ((in_ (prob.GetVector))
					(out_ (prob.PutVector)))
				    "// multiply"
				    (dotimes (i ,N)
				      (let ((v  (aref in_ i)))
					;,(logprint "in" `(v i))
					(setf (aref in i) v)))
				    (do0 ("kernel_hamiltonian<<<blocks,threads,0,stream>>>" out in)
					 (cudaStreamSynchronize stream))
				    (dotimes (i ,N)
				      (let ((v  (aref out i)))
					;,(logprint "out" `(v i))
					(setf (aref out_ i) v))))
				  )))
			    (prob.FindEigenvectors)

			    (dotimes (i 3)
			      ,(logprint "" `(i (prob.Eigenvalue i))))
			    (dotimes (i 1)
			      (dotimes (j ,N)
				(let ((r (* ,dr (+ j 1))))
				 ,(logprint "" `(i r (prob.Eigenvector i j))))))
			    )

			  ;; this error could indicate nan or inf 
			  ;; * On entry to SLASCL parameter number  4 had an illegal value
			  

			  (do0
			   ,@(loop for e in cuda-free collect
				  (cuda `(cudaFree ,e))))))
		      ,(logprint "end main" `())
		      (return 0))))))
  
  (progn
    (progn ;with-open-file
      #+nil (s (asdf:system-relative-pathname 'cl-cpp-generator2
					(merge-pathnames #P"proto2.h"
							 *source-dir*))
	 :direction :output
	 :if-exists :supersede
	 :if-does-not-exist :create)
      #+nil (format s "#ifndef PROTO2_H~%#define PROTO2_H~%~a~%"
		    (emit-c :code `(include <cuda_runtime.h>
					    <cuda.h>
					    <nvrtc.h>)))

      ;; include file
      ;; http://www.cplusplus.com/forum/articles/10627/
      
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 #+nil (progn (format t "emit function declarations for ~a~%" name)
			      (emit-c :code code
				      :hook-defun #'(lambda (str)
						      (format t "~a~%" str))
				      :header-only t))
		 #+nil (emit-c :code code
			 :hook-defun #'(lambda (str)
					 (format s "~a~%" str)
					 )
			 :hook-defclass #'(lambda (str)
					    (format s "~a;~%" str)
					    )
			 :header-only t
			 )
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
		     (format sh "#endif")
		     ))

		 )

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



