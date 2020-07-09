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
  (defparameter *source-dir* #P"example/25_intel_oneapi/source/")
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
	     #+nil("std::setw" 10)
	     #+nil(dot ("std::chrono::high_resolution_clock::now")
		  (time_since_epoch)
		  (count))
					;,(g `_start_time)
	     
	     #+nil(string " ")
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
	      "// nothing"
	      ;(include <chrono>)
	      #+nil (defstruct0 State
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
    `(unless (== cudaSuccess
		 ,code)
       (throw (std--runtime_error (string ,(format nil "~a" (emit-c :code code)))))))
  ;; cp source/*.{h,hpp,cpp} /mnt/c/Users/ThinkPad/Desktop/martin
  ;; run in command line (not powershell)
  ;; C:\Program Files (x86)\inteloneapi\setvars.bat
  ;; cd c:/Users/ThinkPad/Desktop/martin; dgpp vis*.cpp
  ;; https://software.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-programming-model/sample-program.html
  (define-module
      `(main ((_main_version :type "std::string")
	      (_code_repository :type "std::string")
	      (_code_generation_time :type "std::string")
	      )
	     (do0
	      
	      (include <iostream>
		       ;<chrono>
		       ;<cstdio>
		       ;<cassert>
					;<unordered_map>
		       ;;<string>
		       ;;<fstream>
		       ;<thread>
		       ;;<vector>
		       ;;<experimental/iterator>
		       ;;<algorithm>
		       <CL/sycl.hpp>
		       <array>)
	      " "
	      
	      #+nil
	      (do0 "using namespace std::chrono_literals;"
		   (let ((state ,(emit-globals :init t)))
		     (declare (type "State" state))))


	      (defun main (argc argv)
		(declare (values int)
			 (type int argc)
			 (type "char const *const *const" argv))
		(let ((n 1024)
		      (a ("std::array<int,n>"))
		      (b ("std::array<int,n>"))
		      (c ("std::array<int,n>"))
		      )
		  (declare (type "constexpr int" n))
		  (dotimes (i n)
		    (setf (aref a i) i)
		    (setf (aref b i) i)
		    (setf (aref c i) i))
		  (let ((platforms
			 (sycl--platform--get_platforms)))
		    (for-range
		     (&p platforms)
		     ,(logprint "" `((p.get_info<sycl--info--platform--name>)))
		     (let ((devices (p.get_devices)))
		       (for-range
			(&d devices)
			,(logprint "" `((d.get_info<sycl--info--device--name>))))))
		    (let ((s (sycl--default_selector))
			  (q (sycl--queue s))
			  (a_size (sycl--range<1> n))
			  ,@(loop for e in `(a b c) collect
				 `(,(format nil "~a_buf" e)
				    ("sycl::buffer<int,1>"
				     (dot ,e (data))
				     (dot ,e (size)))))
			  (e (q.submit
			      (lambda (h)
				(declare (type sycl--handler& h)
					 (capture "&"))
				(let
				    (,@(loop for (e f) in `((a read)
							     (b read)
							     (c write))
					   collect
					     `(,e
					       (dot ,(format nil "~a_buf" e)
						    (,(format nil "get_access<sycl::access::mode::~a>" f)
						      h)))))
				  (h.parallel_for
				   a_size
				   (lambda (idx)
				     (declare (capture "=")
					      (type "sycl::id<1>" idx))
				     (setf (aref c idx)
					   (+ (aref a idx)
					      (aref b idx))))))))))
		      (e.wait)
		      (progn
			(let ((c (dot c_buf
				      (get_access<sycl--access--mode--read>))))
			  ,(logprint ""
				     `((aref c 0)
				       (aref c 1)
				       (aref c (- n 1)))))))))
		#+nil (do0
		 (setf ,(g `_main_version)
		       (string ,(let ((str (with-output-to-string (s)
					     (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				  (subseq str 0 (1- (length str))))))

		 (setf
		  
		  
		  ,(g `_code_repository) (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/example/19_nvrtc"))
		  
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
		 
		 ,(logprint "start main" `(,(g `_main_version)
					    ,(g `_code_repository)
					    ,(g `_code_generation_time)))
		 
		 
		 ,(logprint "end main" `()))
		(return 0)))))
  
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



