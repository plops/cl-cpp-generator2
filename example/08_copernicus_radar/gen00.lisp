(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  )

(in-package :cl-cpp-generator2)



(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/08_copernicus_radar/source/")
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames #P"run_01_base.c"
											       *source-dir*)))
  
  
  
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)

    (defun emit-globals (&key init)
      (let ((l `(
		 (_start_time double)
		 )))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      "enum {_N_IMAGES=4,_MAX_FRAMES_IN_FLIGHT=2};"
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
		  "#define GLFW_INCLUDE_VULKAN"
		  (include <GLFW/glfw3.h>)
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
			
		  " "
		  )
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header)
	    )
	
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
      `(main ()
	     (do0
	      (let ((state ,(emit-globals :init t)))
		(declare (type State state)))
	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_start_time) (now))
		(run)))))
  (define-module
      `(mmap
	((_window :direction 'out :type VkInstance) )
	(do0
	 (defun init ()
	   ))))
     
   

  (let* ()
					;(write-source *code-file* code)
    ;; we need an empty proto2.h. it has to be written before all c files so that make proto will work
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2
						 (merge-pathnames #P"proto2.h"
								  *source-dir*))
		  `(do0)  (user-homedir-pathname) t)

    (loop for e in (reverse *module*) and i from 0 do
	 (destructuring-bind (&key name code) e
	   (write-source (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (format nil
				  "~a/copernicus_~2,'0d_~a.cpp"
				  *source-dir* i name))
			 code)))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))

		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    
		    (do0
		     
		    " "
		    ,@(loop for e in (reverse *utils-code*) collect
			 e)
		    "#define length(a) (sizeof((a))/sizeof(*(a)))"
					;"#define max(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })"
					;"#define min(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })"
		    "#define max(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a > _b ? _a : _b; })"
		    "#define min(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a < _b ? _a : _b; })"
		    		    
		    " "
		    
		    )
		    " "
		    "#endif"
		    " ")
		  )
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    
		    
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    
    
    
    ;; we need to force clang-format to always have the return type in the same line as the function: PenaltyReturnTypeOnItsOwnLine
					;(sb-ext:run-program "/bin/sh" `("gen_proto.sh"))
    #+nil (sb-ext:run-program "/usr/bin/make" `("-C" "source" "-j12" "proto2.h"))))
 

