(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))




(progn

  (defparameter *source-dir* #P"example/19_nvrtc/source/")
  
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
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))
  (defun cuss (code)
    `(unless (== CUDA_SUCCESS
		 ,code)
       (throw (std--runtime_error (string ,(format nil "~a" (emit-c :code code)))))))
  (defun cuda (code)
    `(unless (== cudaSuccess
		 ,code)
       (throw (std--runtime_error (string ,(format nil "~a" (emit-c :code code)))))))

  (define-module
      `(main ((_main_version :type "std::string")
	      (_code_repository :type "std::string")
	      (_code_generation_time :type "std::string")
	      )
	     (do0
	      "// g++ -march=native -Ofast --std=gnu++20 vis_00_main.cpp -I/media/sdb4/cuda/11.0.1/include/ -L /media/sdb4/cuda/11.0.1/lib -lcudart -lcuda"
	      (include <iostream>
		       <chrono>
		       <cstdio>
		       <cassert>
					;<unordered_map>
		       <string>
		       <fstream>
		       <thread>
		       )
	      (include "vis_02_cu_device.cpp")
	      (include "vis_01_rtc.cpp")

	      "using namespace std::chrono_literals;"
	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))


	      (defun main ()
		(declare (values int))
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
		(let ((dev (CudaDevice--FindByProperties
			    (CudaDeviceProperties--ByIntegratedType false))))
		  (dev.setAsCurrent)
		  (let ((ctx (CudaContext dev))))
		  )
		,(logprint "end main" `())
		(return 0)))))

  
  
  
  (define-module
      `(rtc
	()
	(do0
	 (include <nvrtc.h>
		  <string>
		  <fstream>
		  <streambuf>)
	 (comments "Code c{ <params> };  .. initialize"
		   "Code c = Code::FromFile(fname);  .. load contents of file"
		   "auto& code = c.code() .. get reference to internal string")
	 (defclass Code ()
	   (let ((_code))
	     (declare (type "const std--string" _code)))
	   "public:"
	   (defun Code (...args)
	     (declare (type ARGS&& ...args)
		      (values "template<typename... ARGS> explicit")
		      (construct (_code (space (std--forward<ARGS> args) "...")))))
	   (defun code ()
	     (declare (values "const auto&")
		      (const))
	     (return _code)))
	 (defclass Program ()
	   (let ((_prog))
	     (declare (type nvrtcProgram _prog))))
	 )))
  (define-module
      `(cu_device
	()
	(do0
	 "//  g++ --std=gnu++20 vis_02_cu_device.cpp -I /media/sdb4/cuda/11.0.1/include/"
	 (include <cuda_runtime.h>
		  <cuda.h>)
	 (include <algorithm>
		  <vector>)
	 (defclass CudaDeviceProperties ()
		(let ((_props))
		  (declare (type cudaDeviceProp _props)))
		(defun CudaDeviceProperties (props)
		  (declare (type "const cudaDeviceProp&" props)
			   (construct (_props props))
			   (values explicit)
			   ))
		"public:"
		(defun CudaDeviceProperties (device)
		  (declare (type int device)
			   (values :constructor))
		  (cudaGetDeviceProperties &_props device)
		  (let ((nameSize (/ (sizeof _props.name)
				     (sizeof (aref _props.name 0)))))
		    (setf (aref _props.name (- nameSize 1))
			  (char \\0))))
		(defun FromExistingProperties (props)
		  (declare (type "const cudaDeviceProp&" props)
			   (values "static CudaDeviceProperties"))
		  (return (space CudaDeviceProperties (curly props))))
		(defun ByIntegratedType (integrated)
		  (declare (type bool integrated)
			   (values "static CudaDeviceProperties"))
		  (let ((props (space cudaDeviceProp (curly 0))))
		    (setf props.integrated (? integrated 1 0))
		    ;,(logprint "" `(props))
		    (return (FromExistingProperties props))))
		(defun getRawStruct ()
		  (declare (values "const auto&")
			   (const))
		  (return _props))
		,@(loop for e in `((major)
				   (minor)
				   (integrated :type bool :code (< 0 _props.integrated))
				   (name :type "const char*"))
		     collect
		       (destructuring-bind (name &key code (type "auto")) e
			 `(defun ,name ()
			    (declare (values ,type)
				     (const))
			    ,(if code
				`(return ,code)
				`(return (dot _props ,name)))))))
	 (defclass CudaDevice ()
	   (let ((_device)
		 (_props))
	     (declare (type int _device)
		      (type CudaDeviceProperties _props)))
	   "public:"
	   (defun CudaDevice (device)
	     (declare (type int device)
		      (values explicit)
		      (construct (_device device)
				 (_props device))))
	   (defun handle ()
	     (declare (values "inline CUdevice")
		      (const))
	     (let ((h ))
	       (declare (type CUdevice h))
	       ,(cuss `(cuDeviceGet &h _device))
	       (return h)))
	   (defun FindByProperties (props)
	     (declare (values "static CudaDevice")
		      (type "const CudaDeviceProperties&" props))
	     (let ((device ))
	       (declare (type int device))
	       ,(cuda `(cudaChooseDevice &device (&props.getRawStruct)))
	       (return (space CudaDevice (curly device)))))
	   (defun NumberOfDevices ()
	     (declare (values "static int")
		      )
	     (let ((numDevices 0))
	       (declare (type int numDevices))
	       ,(cuda `(cudaGetDeviceCount &numDevices))
	       (return numDevices)))
	   (defun setAsCurrent ()
	     (cudaSetDevice _device))
	   ,@(loop for e in `((properties :type "const auto &" :code _props)
			      (name :type "const char*" :code (dot (properties)
								   (name))))
		     collect
		       (destructuring-bind (name &key code (type "auto")) e
			 `(defun ,name ()
			    (declare (values ,type)
				     (const))
			    (return ,code))))
	   (defun FindByName (name)
	     (declare (values "static CudaDevice")
		      (type std--string name))
	     
	     (let ((numDevices (NumberOfDevices)))
	       
	       (when (== numDevices 0)
		 (throw (std--runtime_error (string "no cuda devices found"))))
	       (std--transform (name.begin)
			       (name.end)
			       (name.begin)
			       --tolower)
	       (dotimes (i numDevices)
		 (let ((devi (CudaDevice i))
		       (deviName (std--string (devi.name))))
		   #+nil (declare (type CudaDevice (space devi (curly i)))
			    (type std--string (space deviName (curly (devi.name)))))
		   (std--transform (deviName.begin)
				   (deviName.end)
				   (deviName.begin)
				   --tolower)
		   (unless (== std--string--npos (deviName.find name))
		     (return devi))))
	       (throw (std--runtime_error (string "could not find cuda device by name")))))
	   (defun EnumerateDevices ()
	     (declare (values "static std::vector<CudaDevice>")
		      )
	     
	     (let ((res)
		   (n (NumberOfDevices)))
	       (declare (type std--vector<CudaDevice> res))
	       (dotimes (i n)
		 (res.emplace_back i))
	       (return res)))
	   (defun CurrentDevice ()
	     (declare (values "static CudaDevice"))
	     
	     (let ((device)
		   )
	       (declare (type int device))
	       ,(cuda `(cudaGetDevice &device))
	       (return (space CudaDevice (curly device)))))
	   )

	 (defclass CudaContext ()
	   (let ((_ctx))
	     (declare (type CUcontext _ctx)))
	   "public:"
	   (defun CudaContext (device)
	     (declare (type "const CudaDevice&" device)
		      (construct (_ctx nullptr))
		      (values :constructor))
	     ,(cuss `(cuInit 0))
	     ,(cuss `(cuCtxCreate &_ctx 0 (device.handle))))
	   (defun ~CudaContext ()
	     (declare (values :constructor))
	     (when _ctx
	       (cuCtxDestroy _ctx)
	       
	       #+nil(unless (== CUDA_SUCCESS (cuCtxDestroy _ctx))
		 ,(logprint "error when trying to destroy context" `()))))))))
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						      (merge-pathnames #P"proto2.h"
								       *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (format s "#ifndef PROTO2_H~%#define PROTO2_H~%")
      
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 #+nil (progn (format t "emit function declarations for ~a~%" name)
			      (emit-c :code code :hook-defun 
				      #'(lambda (str)
					  (format t "~a~%" str))))
		 (emit-c :code code :hook-defun 
			 #'(lambda (str)
			     (format s "~a~%" str))))

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
      (format s "#endif"))
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
