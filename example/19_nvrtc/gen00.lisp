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
		       <vector>
		       )
	      " "
	      (include
	       <nvrtc.h>
	       <cuda.h>
	       <cuda_runtime.h>)
	      " "
	      (include "vis_03_cu_program.hpp")
	      (include "vis_04_cu_module.hpp")
	      " "
	      (include "vis_02_cu_device.hpp")
	      " "
	      (include "vis_01_rtc.hpp")
	      
	      " "
	      
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
		  (let ((ctx (CudaContext dev))
			(code (Code--FromFile (string "bla.cu")))
			(program (Program (string "myprog") code))
			(kernel (dot (Kernel (string "setKernel"))
				     ("instantiate<float, std::integral_constant<int,10>>")))))
		  
		  (do0
		   (program.registerKernel kernel)
		    (program.compile (curly
				      (;options--
				       GpuArchitecture (dev.properties))
				      (;options--
				       CPPLang ;options--
					       CPP_x17)))
		   
		   (let ((module (Module ctx program))
			 )
		     (kernel.init module program)
		     ))
		  )
		,(logprint "end main" `())
		(return 0)))))

  
  
  
  (define-module
      `(rtc
	()
	(do0
	 (include <nvrtc.h>
		  <cuda.h>
		  <string>
		  <fstream>
		  <streambuf>)

	 (include "vis_04_cu_module.hpp")
	 
	 (include "vis_02_cu_device.hpp")
	 (include "vis_01_rtc.hpp")
	 
	 (comments "Code c{ <params> };  .. initialize"
		   "Code c = Code::FromFile(fname);  .. load contents of file"
		   "auto& code = c.code() .. get reference to internal string")
	 (defclass Code ()
	   (let ((_code))
	     (declare (type "const std::string" _code)))
	   "public:"
	   (defun Code (...args)
	     (declare (type ARGS&& ...args)
		      (values "template<typename... ARGS>") ; explicit
		      ;; who thought of this grammar? this is a mess
		      (construct (_code (space (std--forward<ARGS> args) "...")))))
	   (defun FromFile (name)
	     (declare (type "const std::string&" name)
		      (static)
		      (values "static Code"))
	     (let ((input (std--ifstream name)))
	       (unless (input.good)
		 (throw (std--runtime_error (string "can't read file"))))
	       (input.seekg 0 std--ios--end)
	       (let ((str ;(std--string (input.tellg)  (char "\\0"))
		      ))
		 (declare (type "std::string"  str))
		 (str.reserve (input.tellg))
		 (input.seekg 0 std--ios--beg)
		 (str.assign (std--istreambuf_iterator<char> input)
			     (std--istreambuf_iterator<char>))
		 (return (space Code (curly (std--move str)))))))
	   (defun code ()
	     (declare (values "const auto&")
		      (const))
	     (return _code)))

	 
	 (defclass Header "public Code"
	   (let ((_name))
	     (declare (type "const std::string" _name))
	     )
	   "public:"
	   (defun Header (name ...args)
	     (declare (type "const std::string&" name)
		      (type ARGS&& ...args)
		      (values "template<typename... ARGS>")
		      ;; who thought of this grammar? this is a mess
		      (construct (Code (space (std--forward<ARGS> args) "..."))
				 (_name name))))
	   ,@(loop for e in `(
			      (name :type "const auto&" :code _name))
		     collect
		       (destructuring-bind (name &key code (type "auto")) e
			 `(defun ,name ()
			    (declare (values ,type)
				     (const))
			    (return ,code))))
	   )

	 (progn ;space namespace detail
		(progn
		 (defun BuildArgs (...args)
		   (declare (type "const ARGS&" ...args)
			    (values "template<typename... ARGS> static inline std::vector<void*>"))
		   (return (curly (space
				   (const_cast<void*>
				    ("reinterpret_cast<const void*>" &args))
				   "..."))))
		 (progn ;space "template<typename T>"
		   (defclass (NameExtractor :template "template<typename T>") ()
		     "public:"
			  (defun extract ()
		      (declare (values "static std::string"))
		      (let ((type_name))
			(declare (type std--string type_name))
			(nvrtcGetTypeName<T> &type_name)
		       (return type_name)))))
		 
		 (progn ;space "template<typename T, T y>"
		   (defclass ("NameExtractor<std::integral_constant<T, y>>"
			      :template "template<typename T, T y>") ()
		    "public:"
		    (defun extract ()
		      (declare (values "static std::string"))
		      (return (std--to_string y)))))
		 ))

	 
	 
	 
	 (defclass Kernel ()
	   (let ((_kernel nullptr)
		 (_name))
	     (declare (type CUfunction _kernel)
		      (type std--string _name)))
	   "public:"
	   (defun Kernel (name)
	     (declare (type "const std::string&" name)
		      (values inline)
		      (construct (_name name))))
	   (defclass TemplateParameters ()
	     (let ((_val)
		   (_first true))
	       (declare (type std--string _val)
			(type bool _first))
	       (defun addComma ()
		 (if _first
		     (setf _first false)
		     (setf _val (+ _val (string ",")))))
	       "public:"
	       (defun addValue (val)
		 (declare (values "template<typename T> auto&")
			  (type "const T&" val))
		 (addComma)
		 (setf _val (+ _val (std--string val)))
		 (return *this))
	       (defun addType ()
		 (declare (values "template<typename T> auto&"))
		 (addComma)
		 (setf _val (+ _val (;detail--
				     NameExtractor<T>--extract)))
		 (return *this))
	       (defun "operator()" ()
		 (declare (const)
			  (values "const std::string&"))
		 (return _val))))
	   (defun instantiate (tp)
	     (declare (type "const TemplateParameters&" tp)
		      (values "inline Kernel&"))
	     (setf _name (+ _name
			    (string "<")
			    (tp)
			    (string ">")))
	     (return *this))
	   (defun* instantiate ()
	     (declare (values "template<typename... ARGS> Kernel&"))
	     )
	   (defun name ()
	     (declare (values "const auto&")
		      (const))
	     (return _name))
	   (defun init (m p)
	     (declare (type "const Module&" m)
		      (type "const Program&" p))
	     ,(cuss `(cuModuleGetFunction &_kernel (m.module) (dot p (loweredName *this)
								   (c_str)))))
	   )
	 (do0
	  (defclass CompilationOptions ()
	    (let ((_options)
		  (_chOptions))
	      (declare (type "std::vector<std::string>" _options)
		       (type "mutable std::vector<const char*>" _chOptions)))
	    "public:"
	    (defun insert (op)
	      (declare (type "const std::string&" op))
	      (dot _options (push_back op)))
	    (defun insert (name value)
	      (declare (type "const std::string&" name value))
	      (if (value.empty)
		  (insert name)
		  (dot _options (push_back (+ name (string "=") value)))))
	    (defun insertOptions (p)
	      (declare (type "const T&" p)
		       (values "template<typename T> void"))
	      (insert (p.name) (p.value)))
	    (defun insertOptions (p ...ts)
	      (declare (type "const T&" p)
		       (type "const TS&" ...ts)
		       (values "template<typename T, typename... TS> void"))
	      (insert (p.name) (p.value))
	      (insertOptions ts...)
	      )
	   
	   
	    (defun CompilationOptions (...ts)
	      (declare (type "TS&&" ...ts)
		       (values "template<typename... TS>"))
	      (insertOptions ts...))
	    (setf (CompilationOptions) default)
	    (defun numOptions ()
	      (declare (values auto)
		       (const))
	      (return (_options.size)))
	    (defun options ()
	      (declare (const)
		       (values "const char**")
		       )
	      (dot _chOptions
		   (resize (_options.size)))
	      (std--transform (_options.begin)
			      (_options.end)
			      (_chOptions.begin)
			      (lambda (s)
				(declare (type "const auto&" s))
				(return (s.c_str))))
	      (return (dot _chOptions
			   (data))))
	    )
	  (progn ;space namespace options
		     (progn
		       (defclass GpuArchitecture ()
			 (let ((_arch))
			   (declare (type "const std::string" _arch)))
			 "public:"
			 (defun GpuArchitecture (major minor)
			   (declare (type int major minor)
				    (values :constructor)
				    (construct (_arch (+ (std--string (string "compute_"))
							 (std--to_string major)
							 (std--to_string minor))))))
			 (defun GpuArchitecture (props)
			   (declare (type "const CudaDeviceProperties&" props)
				    (values :constructor)
				    (construct (GpuArchitecture (props.major)
								(props.minor)))))
			 (defun name ()
			   (declare (const)
				    (values auto))
			   (return (string "--gpu-architecture")))
			 (defun value ()
			   (declare (const)
				    (values auto&))
			   (return _arch)))
		       (do0
			(space enum CPPLangVer
			       (curly
				CPP_x11
				CPP_x14
				CPP_x17))
			(defclass CPPLang ()
			  (let ((_version))
			    (declare (type CPPLangVer _version))
			    )
			  "public:"
			  (defun CPPLang (version)
			    (declare (values :constructor)
				     (type CPPLangVer version)
				     (construct (_version version))))
			  (defun name ()
			    (declare (const)
				     (values auto))
			    (return (string "--std")))
			  (defun value ()
			    (declare (const)
				     (values auto))
			    (case _version
			      (CPP_x11 (progn (return (string "c++11"))))
			      (CPP_x14 (progn (return (string "c++14"))))
			      (CPP_x17 (progn (return (string "c++17")))))
			    (throw (std--runtime_error (string "unknown C++ version")))))))))
	 
	 


	 (progn ;space namespace detail
		(progn
		  (defun AddTypesToTemplate (params)
		   (declare (values "static inline void")
			    (type "Kernel::TemplateParameters&" params))
		   )
		  (defun AddTypesToTemplate (params)
		   (declare (values "template<typename T> static inline void")
			    (type "Kernel::TemplateParameters&" params))
		   (params.addType<T>))
		  (defun AddTypesToTemplate (params)
		   (declare (values "template<typename T, typename U, typename... REST> static inline void")
			    (type "Kernel::TemplateParameters&" params))
		   (params.addType<T>)
		   ("AddTypesToTemplate<U, REST...>" params))))

	 (defun "Kernel::instantiate" ()
	     (declare (values "template<typename... ARGS> inline Kernel&"))
	     (let ((tp))
	       (declare (type TemplateParameters tp))
	       (;detail--
		AddTypesToTemplate<ARGS...> tp)
	       (return (instantiate tp))))

	 
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
	 (include "vis_02_cu_device.hpp")
	 (defclass CudaDeviceProperties ()
		(let ((_props))
		  (declare (type cudaDeviceProp _props)))
		(defun CudaDeviceProperties (props)
		  (declare (type "const cudaDeviceProp&" props)
			   (construct (_props props))
			   (values :constructor) ;explicit
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
			   (values CudaDeviceProperties)
			   (static))
		  (return (space CudaDeviceProperties (curly props))))
		(defun ByIntegratedType (integrated)
		  (declare (type bool integrated)
			   (values "CudaDeviceProperties")
			   (static))
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
		      (values :constructor) ; explicit
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
	     (declare (values CudaDevice)
		      (static)
		      (type "const CudaDeviceProperties&" props))
	     (let ((device ))
	       (declare (type int device))
	       ,(cuda `(cudaChooseDevice &device (&props.getRawStruct)))
	       (return (space CudaDevice (curly device)))))
	   (defun NumberOfDevices ()
	     (declare (values int)
		      (static)
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
	     (declare (values CudaDevice)
		      (static)
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
	     (declare (values std--vector<CudaDevice>)
		      (static)
		      )
	     
	     (let ((res)
		   (n (NumberOfDevices)))
	       (declare (type std--vector<CudaDevice> res))
	       (dotimes (i n)
		 (res.emplace_back i))
	       (return res)))
	   (defun CurrentDevice ()
	     (declare (values CudaDevice)
		      (static))
	     
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


  (define-module
      `(cu_program
	()
	(do0
	 
	 (include <cuda_runtime.h>
		  <cuda.h>)
	 (include <algorithm>
		  <vector>)
	 " "
	 (include "vis_01_rtc.hpp")
	 " "
	 (include "vis_03_cu_program.hpp")
	 " "
	 (defclass Program ()
	   (let ((_prog))
	     (declare (type nvrtcProgram _prog)))
	   "public:"
	   (defun Program (name code headers)
	     (declare (type "const std::string&" name)
		      (type "const Code&" code)
		      (type "const std::vector<Header>&" headers)
		      
		      (values :constructor))
	     (let ((nh (headers.size))
		   (headersContent)
		   (headersNames))
	       (declare (type "std::vector<const char*>" headersContent headersNames))
	       (for-range (&h headers)
			  (headersContent.push_back (dot h (code) (c_str)))
			  (headersContent.push_back (dot h (name) (c_str))))
	       ,(rtc `(nvrtcCreateProgram
				  &_prog
				  (dot code (code) (c_str))
				  (name.c_str)
				  (static_cast<int> nh)
				  (? (< 0 nh) (headersContent.data) nullptr)
				  (? (< 0 nh) (headersNames.data) nullptr)))))
	   (defun Program (name code)
	     (declare (type "const std::string&" name)
		      (type "const Code&" code)
		      (construct (Program name code (curly)))
		      (values :constructor)))
	   (defun registerKernel (k)
	     (declare (type "const Kernel&" k)
		      (values "inline void"))
	     ,(rtc `(nvrtcAddNameExpression _prog (dot k
							 (name)
							 (c_str)))))
	   (defun compile (&key (opt (curly)))
	     (declare (type "const CompilationOptions&" opt))
	     (unless (== NVRTC_SUCCESS (nvrtcCompileProgram _prog
							    (static_cast<int> (opt.numOptions))
							    (opt.options)))
	       (let ((logSize))
		 (declare (type std--size_t logSize))
		 (nvrtcGetProgramLogSize _prog &logSize)
		 (let ((log (std--string logSize (char "\\0"))))
		   (nvrtcGetProgramLog _prog (&log.front))
		   (throw (std--runtime_error (log.c_str)))))))
	   (defun PTX ()
	     (declare (values "inline std::string")
		      (const))
	     (let ((size 0))
	       (declare (type std--size_t size))
	       ,(rtc `(nvrtcGetPTXSize _prog &size))
	       (let ((str (std--string size (char "\\0"))))
		 ,(rtc `(nvrtcGetPTX _prog (&str.front)))
		 (return str)))))
	 )))

  (define-module
      `(cu_module
	()
	(do0
	 
	 (include <cuda_runtime.h>
		  <cuda.h>)
	 (include <algorithm>
		  <vector>)
	 " "
	; (include "vis_02_cu_device.hpp")
	 " "
	 (include "vis_03_cu_program.hpp")
	 " "
	 
	 " "
	 (include "vis_04_cu_module.hpp")
	 " "
	 (defclass Module ()
	   (let ((_module))
	     (declare (type CUmodule _module))
	     )
	   "public:"
	   (defun Module (ctx p)
	     (declare (type "const CudaContext&" ctx)
		      (type "const Program&" p)
		      (values :constructor))
	     (cuModuleLoadDataEx &_module
				 (dot p (PTX) (c_str))
				 0 0 0))
	   (defun module ()
	     (declare (values auto)
		      (const))
	     (return _module)))
	 )))


  


  
  
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
