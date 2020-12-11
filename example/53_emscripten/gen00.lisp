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
(defvar *header-file-hashes* (make-hash-table))

(progn
  (defparameter *source-dir* #P"example/53_emscripten/source/")
  
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
      `(progn ;do0
	" "
	#-nolog
	(let ((lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
	      )
	 
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
	      "std::flush")))))
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
	  (unless (or (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
		      (cl-ppcre:scan "base" (string-downcase (format nil "~a" module-name)))
		      (cl-ppcre:scan "mesher_module" (string-downcase (format nil "~a" module-name))))
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
	       (_stdout_mutex :type "std::mutex")
	       )
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			
					;<future>
					; <experimental/future>
		
			)
	       " "
	       (include <SDL2/SDL.h>
			<complex>)
	       " "
	       (do0 
		"#ifdef __EMSCRIPTEN__"
		(include <emscripten.h>)
		"#endif")
	       " "
	  

	    	       "using namespace std::chrono_literals;"
	       " "
	       
	      
	       
	       (let ((state ,(emit-globals :init t)))
		 (declare (type "State" state)))

	    
	       (defun main (argc argv)
		 (declare (type int argc)
			  (type char** argv)
			  (values int))
		 (do0
		  (setf ,(g `_main_version)
			(string ,(let ((str (with-output-to-string (s)
					      (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				   (subseq str 0 (1- (length str))))))

		  (setf
		   ,(g `_code_repository) (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/~a"
							   *source-dir*))
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
		 
	
		  )

		 (let ((width 600)
		       (height 800))
		   (SDL_Init SDL_INIT_VIDEO)
		   (let ((window (static_cast<SDL_Window*> nullptr))
			 (renderer (static_cast<SDL_Renderer*> nullptr))
			 )
		     (SDL_CreateWindowAndRenderer window height
						  SDL_WINDOW_OPENGL
						  &window
						  &renderer)))
		      

		 (return 0)))))
    (define-module
       `(demangle ()
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			
					;<future>
					; <experimental/future>
			
			)

	       " "

	       (include <cxxabi.h>)
	       " "


	       	       
	       "using namespace std::chrono_literals;"

	      
	       " "
	       
	       (defun demangle (name)
		 (declare (type ;"const char*"
			   "const std::string"
			   name)
			  (values "std::string"))
		 (let ((status -4))
		   "std::unique_ptr<char,void(*)(void*)> res {abi::__cxa_demangle(name.c_str(), nullptr,nullptr,&status),std::free};"
		   (if (== 0 status)
		       (return (res.get))
		       (return name))))
	       (defun type_name ()
		 (declare (values "template<class T> std::string"))
		 "typedef typename std::remove_reference<T>::type TR;"
		 "std::unique_ptr<char,void(*)(void*)> own(nullptr,std::free);"
		 "std::string r = (own != nullptr) ? own.get() : typeid(TR).name();"
		 (setf r (demangle r))
		 ,@(loop for (e f) in `(
					(" const" std--is_const<TR>--value)
					(" volatile" std--is_volatile<TR>--value)
					("&" std--is_lvalue_reference<TR>--value)
					("&&" std--is_rvalue_reference<TR>--value))
			 collect
			 `(when ,f
			    (incf r (string ,e))))
		 (return r))
	       )))
    
    
  )
  
  (progn
    
    
    (progn ;with-open-file
      #+nil (s (asdf:system-relative-pathname 'cl-cpp-generator2
					(merge-pathnames #P"proto2.h"
							 *source-dir*))..
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
			(file-h (string-upcase (format nil "~a_H" file)))
			(fn-h (asdf:system-relative-pathname 'cl-cpp-generator2
								      (format nil "~a/~a.hpp"
									      *source-dir* file)))
			
			(code-str (with-output-to-string (sh)
				    (format sh "#ifndef ~a~%" file-h)
				    (format sh "#define ~a~%" file-h)
			 
				    (emit-c :code code
								    :hook-defun #'(lambda (str)
										    (format sh "~a~%" str))
								    :hook-defclass #'(lambda (str)
										       (format sh "~a;~%" str))
					    :header-only t)
				    (format sh "#endif")))
			(fn-hash (sxhash fn-h))
			(code-hash (sxhash code-str)))
		   (multiple-value-bind (old-code-hash exists) (gethash fn-hash *header-file-hashes*)
		     (when (or (not exists)
			       (/= code-hash old-code-hash)
			       (not (probe-file fn-h)))
		       ;; store the sxhash of the header source in the hash table
		       ;; *header-file-hashes* with the key formed by the sxhash of the full
		       ;; pathname
		       (setf (gethash fn-hash *header-file-hashes*) code-hash)
		       (format t "~&write header: ~a fn-hash=~a ~a old=~a~%" fn-h fn-hash code-hash old-code-hash
			       )
		       (with-open-file (sh fn-h
					   :direction :output
					   :if-exists :supersede
					   :if-does-not-exist :create)
			 (format sh "#ifndef ~a~%" file-h)
			 (format sh "#define ~a~%" file-h)
			 
			 (emit-c :code code
				 :hook-defun #'(lambda (str)
						 (format sh "~a~%" str))
				 :hook-defclass #'(lambda (str)
						    (format sh "~a;~%" str))
				 :header-only t)
			 (format sh "#endif"))
		       (sb-ext:run-program "/usr/bin/clang-format"
					   (list "-i"  (namestring fn-h)))))))
	       (progn
		#+nil (format t "emit cpp file for ~a~%" name)
		(write-source (asdf:system-relative-pathname
			       'cl-cpp-generator2
			       (format nil
				       "~a/vis_~2,'0d_~a.~a"
				       *source-dir* i name
				       (if cuda
					   "cu"
					   "cpp")))
			      code)))))
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
			     ;<array>
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
		     (include ;<thread>
			      <mutex>
			     ;<queue>
			     ;<condition_variable>
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
		    " "))
    (with-open-file (s "source/CMakeLists.txt" :direction :output
					       :if-exists :supersede
					       :if-does-not-exist :create)
      ;; run emcmake cmake .. && make
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	(out "cmake_minimum_required( VERSION 3.15 )")
	(out "project( mytest LANGUAGES CXX )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set( CMAKE_CXX_STANDARD 14 )")

	;; https://stackoverflow.com/questions/61590519/how-to-use-emscripten-ports-sdl2-and-freetype-with-cmake
	(out "if( ${CMAKE_SYSTEM_NAME} MATCHES \"Emscripten\" )")
	(out "  set( USE_FLAGS \"-s USE_SDL=2\" )")
	(out "  set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${USE_FLAGS} )")
	(out "  set( CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${USE_FLAGS} )")
	(out "  set( CMAKE_EXECUTABLE_SUFFIX .html )")
	(out "else()")
	(out "find_package( SDL2 REQUIRED )")
	(out "endif()")
	
					;(out "set( CMAKE_CXX_FLAGS )")
	(out "set( SRCS ~{~a~^~%~} )"	;(directory "source/*.cpp")
	     `(vis_00_base.cpp
	       vis_01_demangle.cpp))
	(out "add_executable( mytest ${SRCS} )")
	)
      )))



