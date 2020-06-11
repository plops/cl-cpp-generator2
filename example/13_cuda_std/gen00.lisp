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
		   ;;-I/opt/cuda/targets/x86_64-linux/include/cuda/std/detail/ -I/opt/cuda/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/
		   "// /opt/cuda/bin/nvcc /media/sdb4/cuda/b/cuda_nvcc/bin/nvcc custd_00_cuda_main.cu --gpu-architecture=compute_75 --gpu-code=compute_75 --use_fast_math  -I/opt/cuda/include/  --std=c++14 -O3 -g -Xcompiler=-march=native --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0 -I/media/sdb4/cuda/b/cuda_cudart/targets/x86_64-linux/include/ "
		   "// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/cwe21285.pdf p. 338"
		   "// https://on-demand.gputechconf.com/supercomputing/2019/video/sc1942-the-cuda-c++-standard-library/"
		   "// https://x.momo86.net/?p=107 .. japanese article, maybe string isn't supported yet"
		   "// https://on-demand.gputechconf.com/supercomputing/2019/video/sc1942-the-cuda-c++-standard-library/"
		   (include ;<cstdio>
			    <cuda/std/atomic>
			    
			    ;"/opt/cuda/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/string_view"
			    ;<cuda/std/detail/libcxx/include/__config>
			    <cuda/std/detail/libcxx/include/string_view>
			    )
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
	      

	      "using namespace std::chrono_literals;"
	      ;"using namespace cuda;"
	      

	      (space "struct trie"
		     (progn
		       ;; indexed by alphabet
		       (space "struct ref"
			      (progn 
				(let ((ptr #+host nullptr
					   #-host (ATOMIC_VAR_INIT nullptr))
				      (flag ATOMIC_FLAG_INIT))
				  (declare (type #+host trie*
						 #-host "cuda::std::atomic<trie*>" ptr)
					   (type "cuda::std::atomic_flag" flag))))
			      "next[26]")
		       
		       #+host (let ((count 0) ;; mapped value for this position of the trie
			     )
				(declare (type int count)))
		       #-host (let ((count (ATOMIC_VAR_INIT 0)))
				(declare (type "cuda::std::atomic<int>" count)))
		       (space "__host__ __device__"
			(defun insert (input &bump)
			  (declare (type cuda--std--string_view
				    ;char*
					 input)
				   (type #+host trie*
					 #-host cuda--std--atomic<trie*>
					 &bump)
				   )
			  (let ((n this)) ;; n .. current position in trie
			    (for-range (pc input)
				       (let ((index (index_of pc)))
					 (declare (type "auto const" index))
					 (when (== index -1) ;; end of word
					   (when (!= n this) ;; word isn't empty
					     (incf n->count)
					     (setf n this) ;; reset position
					     )
					   continue)
					 (when (== (dot (aref n->next index)
							ptr)
						   nullptr) ;; node doesnt exist
					   (incf bump) ;; allocate new node
					   (setf (dot (aref n->next index)
						      ptr)
						 bump))
					 ;; advance position
					 (setf n (dot (aref n->next index)
						      ptr))
					 )))))))

	      (defun index_of (c)
		(declare (type char c)
			 (values int))
		(when (<= (char "a") c (char "z"))
		  (return (- c (char "a"))))
		(when (<= (char "A") c (char "Z"))
		  (return (- c (char "A"))))
		(return -1))
	      
	      	      
	      (defun main ()
		(declare (values int))
		
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

