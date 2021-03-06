(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre")
     (ql:quickload "cl-change-case")) 

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))
(defvar *header-file-hashes* (make-hash-table))



(progn
  (defparameter *source-dir* #P"example/60_wrldtmpl/source/")
  
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
      `(progn				;do0
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
		      (cl-ppcre:scan "base" (string-downcase (format nil "~a" module-name))))
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
		)
	       (do0
		(split-header-and-code
		 (do0 (comments "header")
		      (do0 (include <iostream>
				    <chrono>
				    <thread>
				    )
			   " "
			   ))
		 (do0 (comments "implementation")
		      (include <vis_00_base.hpp>)))

	       
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
		 

		  (return 0)))))

    (define-module
	`(demangle ()
		   (do0
		    (include <iostream>
			     <chrono>
			     <thread>
			     )

		    " "

		    (include <cxxabi.h>)
		    " "

	       
		    "using namespace std::chrono_literals;"

		    (defun demangle (name)
		      (declare (type	;"const char*"
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
		      (return r)))))
    (define-module
	`(surface95
	  ()
		  (do0
		   (include <iostream>
			    <chrono>
			    <thread>
			
			    )

		   " "

		   (split-header-and-code
		    (do0 (comments "header")
			 (comments "http://freeimage.sourceforge.net")
			 (include <FreeImage.h>)
			 (include <vis_03_memory.hpp>)
			 )
		    (do0 (comments "implementation")
			 "static char s_Font[51][5][6];"
			 "static bool fontInitialized = false;"
			 "static int s_Transl[256];"
		     
			 ))
	       
		   (defclass Surface ()
		     "enum { OWNER = 1 };"
		     "public:"
		     (defmethod Surface (w h a_Buffer)
		       (declare (type int w h)
				(type uint* a_Buffer)
				(construct (buffer b) (width w) (height h))
				(values :constructor))
		       )
		     (defmethod Surface (w h)
		       (declare (type int w h)
			    
				(values :constructor))
		       (setf buffer (static_cast<uint*> (MALLOC64 (* w h (sizeof uint))))))
		     (defmethod Surface (file)
		       (declare (type "const char*" file)
				(construct (buffer 0) (width 0) (height 0))
				(values :constructor))
		       (let ((f (fopen file (string "rb"))))
			 (unless f
			   ,(logprint "file not found" `(file)))
			 (fclose f)
			 (LoadImage file)))
		     (defmethod ~Surface ()
		       (declare  (values :constructor))
		       (FREE64 buffer))
		     (defmethod InitCharSet ()
		       ,(let ((symbols "abcdefghijklmnopqrstuvwxyz0123456789!?:=.-() #'*/")
			      (l `((":ooo:" "o:::o" "ooooo" "o:::o" "o:::o" )
				   ("oooo:" "o:::o" "oooo:" "o:::o" "oooo:" )
				   (":oooo" "o::::" "o::::" "o::::" ":oooo" )
				   ("oooo:" "o:::o" "o:::o" "o:::o" "oooo:" )
				   ("ooooo" "o::::" "oooo:" "o::::" "ooooo" )
				   ("ooooo" "o::::" "ooo::" "o::::" "o::::" )
				   (":oooo" "o::::" "o:ooo" "o:::o" ":ooo:" )
				   ("o:::o" "o:::o" "ooooo" "o:::o" "o:::o" )
				   ("::o::" "::o::" "::o::" "::o::" "::o::" )
				   (":::o:" ":::o:" ":::o:" ":::o:" "ooo::" )
				   ( "o::o:" "o:o::" "oo:::" "o:o::" "o::o:")
				   ( "o::::" "o::::" "o::::" "o::::" "ooooo")
				   ( "oo:o:" "o:o:o" "o:o:o" "o:::o" "o:::o")
				   ( "o:::o" "oo::o" "o:o:o" "o::oo" "o:::o")
				   ( ":ooo:" "o:::o" "o:::o" "o:::o" ":ooo:")
				   ( "oooo:" "o:::o" "oooo:" "o::::" "o::::")
				   ( ":ooo:" "o:::o" "o:::o" "o::oo" ":oooo")
				   ( "oooo:" "o:::o" "oooo:" "o:o::" "o::o:")
				   ( ":oooo" "o::::" ":ooo:" "::::o" "oooo:")
				   ( "ooooo" "::o::" "::o::" "::o::" "::o::")
				   ( "o:::o" "o:::o" "o:::o" "o:::o" ":oooo")
				   ( "o:::o" "o:::o" ":o:o:" ":o:o:" "::o::")
				   ( "o:::o" "o:::o" "o:o:o" "o:o:o" ":o:o:")
				   ( "o:::o" ":o:o:" "::o::" ":o:o:" "o:::o")
				   ( "o:::o" "o:::o" ":oooo" "::::o" ":ooo:")
				   ( "ooooo" ":::o:" "::o::" ":o:::" "ooooo")
				   ( ":ooo:" "o::oo" "o:o:o" "oo::o" ":ooo:")
				   ( "::o::" ":oo::" "::o::" "::o::" ":ooo:")
				   ( ":ooo:" "o:::o" "::oo:" ":o:::" "ooooo")
				   ( "oooo:" "::::o" "::oo:" "::::o" "oooo:")
				   ( "o::::" "o::o:" "ooooo" ":::o:" ":::o:")
				   ( "ooooo" "o::::" "oooo:" "::::o" "oooo:")
				   ( ":oooo" "o::::" "oooo:" "o:::o" ":ooo:")
				   ( "ooooo" "::::o" ":::o:" "::o::" "::o::")
				   ( ":ooo:" "o:::o" ":ooo:" "o:::o" ":ooo:")
				   ( ":ooo:" "o:::o" ":oooo" "::::o" ":ooo:")
				   ( "::o::" "::o::" "::o::" ":::::" "::o::")
				   ( ":ooo:" "::::o" ":::o:" ":::::" "::o::")
				   ( ":::::" ":::::" "::o::" ":::::" "::o::")
				   ( ":::::" ":::::" ":ooo:" ":::::" ":ooo:")
				   ( ":::::" ":::::" ":::::" ":::o:" "::o::")
				   ( ":::::" ":::::" ":::::" ":::::" "::o::")
				   ( ":::::" ":::::" ":ooo:" ":::::" ":::::")
				   ( ":::o:" "::o::" "::o::" "::o::" ":::o:")
				   ( "::o::" ":::o:" ":::o:" ":::o:" "::o::")
				   ( ":::::" ":::::" ":::::" ":::::" ":::::")
				   ( "ooooo" "ooooo" "ooooo" "ooooo" "ooooo")
				   ( "::o::" "::o::" ":::::" ":::::" ":::::")
				   ( "o:o:o" ":ooo:" "ooooo" ":ooo:" "o:o:o")
				   ( "::::o" ":::o:" "::o::" ":o:::" "o::::"))
				 ))
			  `(do0
		
			    ,@(loop for e in l and i from 0 collect
				    `(SetChar ,i ,@(mapcar (lambda (x) `(string ,x)) e)))
			    (let ((c (string ,symbols)))
			      (declare (type (array char ,(length symbols)) c))
			      (dotimes (i 256)
				(setf (aref s_Transl i) 45))
			      (dotimes (i ,(length symbols))
				(setf (aref s_Transl ("static_cast<unsigned char>" (aref c i))) i)
				)))
			  ))
		     (defmethod SetChar (c1 c2 c3 c4 c5)
		       (declare (type "const char*" c1 c2 c3 c4 c5))
		       ,@(loop for i from 1 upto 5 collect
			       `(strcpy (aref s_Font c ,i)
					,(format nil "c~a" i))))
		     (defmethod Print (tt x1 y1 c)
		       (declare (type "const char*" tt)
				(type int x1 y1)
				(type uint c))
		       (unless fontInitialized
			 (InitCharset)
			 (setf fontInitialized true))
		       (let ((tt (+ buffer x1 (* y1 width))))
			 (dotimes (i (static_cast<int> (strlen s)))
			   (let ((pos 0))
			     (if (<= "'A'" (aref s i) "'Z'")
				 (setf pos (aref s_Transl
						 ("static_cast<unsigned short>" (- (aref s i)
										   (- "'A'" "'a'")))))
				 (setf pos (aref s_Transl
						 ("static_cast<unsigned short>" (aref s i)))))
			     (let ((a tt)
				   (u ("static_cast<const char*>" (aref s_Font pos))))
			       (dotimes (v 5)
				 (dotimes (h 5)
				   (when (== (char o)
					     "*u++")
				     (setf (deref (+ a h))
					   c
					   (deref (+ a h width) ) 0)
				     ))
				 (incf u)
				 (incf a width))))
			   (incf tt 6)))
		   
		       )
		 
		     (defmethod Clear (c)
		       (declare (type uint c)
			    
				)
		       (let ((s (* width height)))
			 (declare (type "const int" s))
			 (dotimes (i s)
			   (setf (aref buffer i) c))))
		     #+nil (defmethod Line (x1 y1 x2 y2 c)
			     (declare (type uint c)
				      (type float x1 y1 x2 y2)
				      ))
		     #+nil (defmethod Plot (x y c)
			     (declare (type uint c)
				      (type int x y)
				      ))
		     (defmethod LoadImage (file)
		       (declare (type "const char*" file)
				)
		       (let ((fif FIF_UNKNOWN))
			 (declare (type FREE_IMAGE_FORMAT fif))
			 (setf fif (FreeImage_GetFileType file 0))
			 (when (== FIF_UNKNOWN fif)
			   (setf fif (FreeImage_GetFIFFromFilename file)))
			 (let ((tmp (FreeImage_Load fif file))
			       (dib (FreeImage_ConvertTo32Bits tmp)))
			   (FreeImage_Unload tmp)
			   (let ((width (FreeImage_GetWidth dib))
				 (height (FreeImage_GetHeight dib))
				 (buffer (static_cast<uint*> (MALLOC64 (* width height (sizeof uint))))))
			     (dotimes (y height)
			       (let ((line (FreeImage_GetScanLine dib (+ height -1 -y))))
				 (memcpy (+ buffer (* y width))
					 line
					 (* width (sizeof uint)))))
			     (FreeImage_Unload dib)))))
		     (defmethod CopyTo (dst a_X a_Y)
		       (declare (type Surface* dst)
				(type int a_X a_Y)
				)
		       (let ((dst d->buffer)
			     (src buffer))
			 (when (and src dst)
			   (let ((sw width)
				 (sh height)
				 (w d->width)
				 (h d->height))
			     (when (< w (+ sw x))
			       (setf sw (- w x)))
			     (when (< h (+ sh y))
			       (setf sh (- h y)))
			     (when (< x 0)
			       (decf src x)
			       (incf sw x)
			       (setf x 0))
			     (when (< y 0)
			       (decf src (* sw y))
			       (incf sh y)
			       (setf y 0))
			     (when (and (< 0 sw)
					(< 0 sh))
			       (incf dst (+ x (* w y)))
			       (dotimes (y sh)
				 (memcpy dst src (* 4 sw))
				 (incf dst w)
				 (incf src sw)))))))
		     (defmethod Box (x1 y1 x2 y2 color)
		       (declare (type int x1 y1 x2 y2)
				(type uint color)))
		     (defmethod Bar (x1 y1 x2 y2 color)
		       (declare (type uint color)
				(type int x1 y1 x2 y2)
				))
		     "uint* buffer;"
		     "int width;"
		     "int height;"))))

    (define-module
	`(memory ()
		 (do0
		  (split-header-and-code
		   (do0 (comments "header")
			"#define ALIGN( x ) __attribute__( (aligned( x ) ) )"
			"#define MALLOC64( x ) ((x)==0?0:aligned_alloc(64,(x)))"
			"#define FREE64(x) free(x)"
			,@(loop for e in `((8 int2 x y)
					   (8 uint2 x y)
					   (8 float2 x y)
					   (16 int3 x y z dummy)
					   (16 uint3 x y z dummy)
					   (16 float3 x y z dummy)
					   (16 int4 x y z w)
					   (16 uint4 x y z w)
					   (16 float4 x y z w)
					   (4 uchar4 x y z w)
					   )
				collect
				(destructuring-bind (bytes name &rest vars) e
				  (let ((type (multiple-value-bind (str found)
						  (cl-ppcre:scan-to-strings "(.*)[0-9]"
									    (format nil "~a" name))
						(aref found 0))))
				    (format nil "struct ALIGN( ~a ) ~a { ~a ~{~a~^, ~}; };"
					    bytes name type vars))))
			)
		   (do0 (comments "implementation"))))))

    (defun gl-upcase (str)
      (string-upcase (cl-change-case:snake-case (format nil "gl-~a" str))))
    (defun tex-param (e )
      ;; tex-param `(texture-min-filter nearest)
      (destructuring-bind (key value &key (target `texture-2d)) e
	`(glTexParameteri ,(gl-upcase target)
			  ,(gl-upcase key)
			  ,(gl-upcase value))))
    (define-module
	`(gl_texture ()
		     (do0
		      (split-header-and-code
		       (do0 (comments "header")
			    )
		       (do0 (comments "implementation")))
		      (defclass GLTexture ()
			"public:"
			"enum { DEFAULT=0, FLOAT=1, INITTARGET=2 };"
			(defmethod GLTexture (w h &key (type DEFAULT))
			  (declare (type uint w h type)
				   (construct (width w)
					      (height h))
				   (values :constructor))
			  (glGenTextures 1 &ID)
			  (glBindTextures GL_TEXTURE_2D ID)
			  (case type
			    (DEFAULT
			     (glTexImage2D GL_TEXTURE_2D 0 GL_RGB width height 0 GL_BGR GL_UNSIGNED_BYTE 0)
			     ,(tex-param `(min_filter nearest))
			     ,(tex-param `(mag_filter nearest))
			 
			     )
			    (INITTARGET
			     ,(tex-param `(texture-wrap-s clamp-to-edge))
			     ,(tex-param `(texture-wrap-t clamp-to-edge))
			     ,(tex-param `(min_filter nearest))
			     ,(tex-param `(mag_filter nearest))
			     (glTexImage2D GL_TEXTURE_2D 0 GL_RGBA8 width height 0 GL_RGBA GL_UNSIGNED_BYTE 0)
			     )
			    (FLOAT
			     ,(tex-param `(texture-wrap-s clamp-to-edge))
			     ,(tex-param `(texture-wrap-t clamp-to-edge))
			     ,(tex-param `(min_filter nearest))
			     ,(tex-param `(mag_filter nearest))
			 
			     (glTexImage2D GL_TEXTURE_2D 0 GL_RGBA32F width height 0 GL_RGBA FLOAT 0)
			     ))
			  (glBindTexture GL_TEXTURE_2D 0)
			  (CheckGL))
			(defmethod ~GLTexture ()
			  (declare 
			   (values :constructor))
			  (glDeleteTextures 1 &ID)
			  (CheckGL))
			(defmethod Bind (&key (slot 0))
			  (declare (type "const uint" slot))
			  (glActiveTexture (+ ,(gl-upcase 'texture0)
					      slot)
					   )
			  (glBindTexture GL_TEXTURE_2D ID)
			  (CheckGL))
			(defmethod CopyFrom (src)
			  (declare (type Surface* src))
			  (do0
			   (glBindTexture GL_TEXTURE_2D ID)
			   (glGetTexImage GL_TEXTURE_2D 0 GL_RGBA width height 0 GL_RGBA GL_UNSIGNED_BYTE src->buffer)
			   (CheckGL)))
			(defmethod CopyTo (dst)
			  (declare (type Surface* dst))
			  (do0
			   (glBindTexture GL_TEXTURE_2D ID)
			   (glGetTexImage GL_TEXTURE_2D 0 GL_RGBA GL_UNSIGNED_BYTE dst->buffer)
			   (CheckGL)))q
			   "GLuint ID = 0;"
			   "uint width=0;"
			   "uint height=0;"))))

    (defun defmethods (&key defs pre post)
      ;; helper function to define a bunch of methods at once
      ;; it makes it easier to create type definitions for parameters
      ;;
      ;; defs ::= [<name> [params] [decl] [return] [code] [nopre] [nopost]]*
      ;; param ::= [<var-name> [types]* [:default <default>]]*
      ;; types will be concatenated with space
      ;; if no types is given, the previous type is used
      ;; decl conveys arbitrary declarations like: ((construct (a 3))

      ;;
      ;; :default keyword separates type declarations from default value in parameter definition
      ;;
      
      ;; pre and post contain code sections that will be inserted in the
      ;; beginning or end of the methods. this feature can be disabled per method
      ;; with the flags nopre and nopost.
      ;;
      ;;
      ;; returns a list of defmethod s-expressions
      (loop for def in defs
	    collect
	    (destructuring-bind (name params &key return code decl nopre nopost) def
	      `(defmethod ,name (,@(let ((key-occurred nil))
				     (loop for param in params
					   appending
					   (destructuring-bind (var-name &rest rest) param
					     #+nil `(,var-name)
					     (if (position :default  rest)
						 (if key-occurred
						     `((,var-name ,(cadr (subseq rest (position :default rest)))))
						     (progn
						       (setf key-occurred t)
						       `(&key (,var-name ,(cadr (subseq rest (position :default rest))))
							     )))
						 `(,var-name))))))
		 (declare ,@(let ((old-type nil))
			      (loop for param in params
				    collect
				    (destructuring-bind (var-name &rest rest) param
				      (when rest
					(let ((types (subseq rest 0 (position :default rest))))
					  (setf old-type (format nil "~{~a ~}" types))))
				      #+nil (setf old-type (format nil "~{~a ~}" (subseq rest
										   0
										   (position :default rest))))
				      `(type ,old-type ,var-name))))
			  (values ,return))
		 ,(unless nopre
		    pre)
		 ,code
		 ,(unless nopost
		    post))))
      )
    (defun defuns (&key defs pre post)
      ;; same as defmethods but for defun
      (loop for def in defs
	    collect
	    (destructuring-bind (name params &key return code decl nopre nopost) def
	      `(defun ,name (,@(loop for param in params
				     collect
				     (destructuring-bind (var-name &rest rest) param
				       var-name)))
		 (declare ,@(let ((old-type nil))
			      (loop for param in params
				    collect
				    (destructuring-bind (var-name &rest rest) param
				      (when rest
					(setf old-type (format nil "~{~a ~}" rest)))
				      `(type ,old-type ,var-name))))
			  (values ,return))
		 ,(unless nopre
		    pre)
		 ,code
		 ,(unless nopost
		    post)))))

    (defun lassert (assertion)
      ;; log msg and parameters if condition fails
      ;; assertion ::= condition [param] [msg] 
      (destructuring-bind (condition &key param msg) assertion
	`(when ,condition
	   ,(logprint msg param))))
    
    (define-module
	`(gl_shader ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")))
		     "class mat4;"
		     (defclass Shader ()
		       "public:"
		       #+nil
		       (defmethod Shader (vfile pfile fromString)
			 (declare (type "const char*" vfile pfile)
				  (type bool fromString)
				  (construct)
				  (values :constructor))
			 )
		       #+nil
		       (defmethod ~Shader ()
			 (declare 
			  (values :constructor))
			 )

		       ,@(defmethods
			     :post `(CheckGL)
			   :defs
			   `((Shader  ((vfile const char*)
				       (pfile)
				       (fromString bool))
				      :return :constructor
				      :code (if fromString
						(Compile vfile pfile)
						(Init vfile pfile))
				      :nopost t)
			     (~Shader ()
				      :return :constructor
				      :code (do0
					     (glDetachShader ID pixel)
					     (glDetachShader ID vertex)
					     (glDeleteShader pixel)
					     (glDeleteShader vertex)
					     (glDeleteProgram ID)
					;(CheckGL)
					     ))
			     (Init  ((vfile const char*)
				     (pfile))
				    :nopost t
				    :code
				    (let ((vsText (TextFileRead vfile))
					  (fsText (TextFileRead pfile)))
				      ,(lassert `((== 0 (vsText.size))
						  :param (vfile)
						  :msg "File not found"))
				      ,(lassert `((== 0 (fsText.size))
						  :param (pfile)
						  :msg "File not found"))
				      (let ((vertexText (vsText.c_str))
					    (fragmentText (fsText.c_str)))
					(Compile vertexText fragmentText))))
			     (Compile  ((vtext const char*)
					(ftext))
				       :code
				       (let ((vertex (glCreateShader GL_VERTEX_SHADER))
					     (pixel (glCreateShader GL_FRAGMENT_SHADER)))

					 ,@(loop for (name code) in `((vertex &vtext)
								      (pixel &ftext))
						 collect
						 `(do0 (glShaderSource ,name 1 ,code 0)
						       (glCompileShader ,name)
						       (CheckShader ,name vtext ftext)))
					 (setf ID (glCreateProgram))
					 ,@(loop for e in `(vertex pixel)
						 collect
						 `(glAttachShader ID ,e))
					 ,@(loop for e in `(pos tuv)
						 and i from 0 
						 collect
						 `(glBindAttribLocation ID ,i (string ,e)))
					 (glLinkProgram ID)
					 (glCheckProgram ID vtext ftext)
					;(CheckGL)
				      
					 ))
			     (Bind () :code (do0
					     (glUseProgram ID)
					;(CheckGL)
					     ))
			     (Unbind () :code
				     (do0
				      (glUseProgram 0)
					;(CheckGL)
				      )
				     )
			     (SetInputTexture  ((slot uint)
						(name const char*)
						(texture GLTexture*))
					       :code
					       (do0
						(glActiveTexture (+ GL_TEXTURE0 slot) )
						(glBindTexture GL_TEXTURE_2D texture->ID)
						(glUniform1i (glGetUniformLocation ID name)
							     slot)
						(CheckGL)))
			     (SetInputMatrix  ((name const char*)
					       (matrix const mat4&))
					      :code
					      (let ((data ("static_cast<const GLfloat*>"
							   &matrix)))
						(glUniformMatrix4fv (glGetUniformLocation ID name)
								    1
								    GL_FALSE
								    data)
					;(CheckGL)
						))
			     (SetFloat  ((name const char*)
					 (v const float))
					:code
					(glUniform1f (glGetUniformLocation ID name)
						     v))
			     (SetInt  ((name const char*)
				       (v const int))
				      :code
				      (glUniform1i (glGetUniformLocation ID name)
						   v))
			     (SetUInt  ((name const char*)
					(v const uint))
				       :code
				       (glUniform1ui (glGetUniformLocation ID name)
						     v))
			     ))
		       "uint vertex = 0, pixel = 0, ID =0;"
		       ))))

    (define-module
	`(gl_helper ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   "#define CheckGL() {_CheckGL( __FILE__, __LINE__ ); }")
		      (do0 (comments "implementation")))
		     ,@(defuns
			   :defs
			   `((_CheckGL  ((f const char*)
					 (l int))
					:code (let ((err (glGetError)))
						(unless (== GL_NO_ERROR err)
						  (let ((errStr (string "UNKNOWN ERROR")))
						    (case err
						      ,@(loop for (e f) in `((#x500 "INVALID ENUM")
									     (#x502 "INVALID OPERATION")
									     (#x501 "INVALID VALUE")
									     (#x506 "INVALID FRAMEBUFFER OPERATION")
									     )
							      collect
							      `(,e (setf errStr (string ,f)))
							      ))
						    ,(logprint "gl error" `(err errStr f l)))))
					)
			     (CreateVBO ((data const GLfloat*)
					 (size const uint))
					:return GLuint
					:code (let ((id ))
						(declare (type GLuint id))
						(glGenBuffers 1 &id)
						(glBindBuffer GL_ARRAY_BUFFER id)
						(glBufferData GL_ARRAY_BUFFER size data GL_STATIC_DRAW)
						(CheckGL)
						(return id)))
			     (BindVBO  ((idx const uint)
					(N)
					(id cont GLuint))
				       :code
				       (do0
					(glEnableVertexAttribArray idx)
					(glBindBuffer GL_ARRAY_BUFFER id)
					(glVertexAttribPointer idx N GL_FLOAT GL_FALSE 0 nullptr)
					(CheckGL)))
			     (CheckShader  ((shader GLuint)
					    (vshader const char*)
					    (fshader))
					   :code
					   (let ((buffer))
					     (declare (type (array char 1024) buffer))
					     (memset buffer 0 (sizeof buffer))
					     (let ((length 0))
					       (declare (type GLsizei length))
					       (glGetShaderInfoLog shader (sizeof buffer)
								   &length buffer)
					       (CheckGL)
					       ,(lassert `((not (and (< 0 length)
								     (strstr buffer (string "ERROR"))))
							   :msg "shader compile error"
							   :param (buffer))))))
			     (CheckProgram ((id GLuint)
					    (vshader const char*)
					    (fshader))
					   :code
					   (let ((buffer))
					     (declare (type (array char 1024) buffer))
					     (memset buffer 0 (sizeof buffer))
					     (let ((length 0))
					       (declare (type GLsizei length))
					       (glGetProgramInfoLog id (sizeof buffer)
								    &length buffer)
					       (CheckGL)
					       ,(lassert `((< length 0)
							   :msg "shader compile error"
							   :param (buffer))))))
			     (DrawQuad ()
				       :code
				       (let ((vao 0))
					 (declare (type
						   "static GLuint" vao))
					 (unless vao
					   (let ((verts (curly -1 1 0
							       1 1 0
							       -1 -1 0
							       1 1 0
							       -1 -1 0
							       1 -1 0))
						 (uvdata (curly 0 0
								1 0
								0 1
								1 0
								0 1
								1 1
								))
						 (vertexBuffer (CreateVBO verts (sizeof verts)))
						 (UVBuffer (CreateVBO uvdata (sizeof uvdata))))
					     (declare (type (array
							     "static const GLfloat"
							     (* 3 6)) verts)
						      (type (array "static const GLfloat" (* 2 6) uvdata)))
					     (glGenVertexArray 1 &vao)
					     (glBindVertexArray vao)
					     (BindVBO 0 3 vertexBuffer)
					     (BindVBO 1 2 UVBuffer)
					     (glBindVertexArray 0)
					     (CheckGL)))
					 (glBindVertexArray vao)
					 (glDrawArrays GL_TRIANGLES 0 6)
					 (glBindVertexArray 0))))))))
    (define-module
	`(job ()
	      (do0
	       (split-header-and-code
		(do0 (comments "header"))
		(do0 (comments "implementation")))
	       ;; hmm this is actually all windows code
	       (defclass Job ()
		 "public:"
		 ;; pure virtual functions not 
		 (defmethod Main ()
		   (declare (virtual)))
		 "protected:"
		 "friend class JobThread;"
		 (defmethod RunCodeWrapper ()))
	       (defclass JobThread ()
		 "public:"
		 ,@(defmethods
		    :defs
		    `((CreateAndStartThread ((threadId unsigned int)))
		      (WaitForThreadToStop ())
		      (Go ())
		      (BackgroundTask ())
		      ))
		 "HANDLE m_GoSignal, m_ThreadHandle;"
		 "int m_ThreadID;"
		 )
	       (defclass JobManager ()
		 #+nil
		 (do0 "protected:"
		  (defmethod JobManager (numThreads)
		    (declare (type "unsigned int" numThreads))))
		 ,@(defmethods
		       :defs
		       `((~JobManager ())
			 (CreateJobManager ((numThreads unsigned int))
					   :return "static void")
			 (GetJobManager () :return "static JobManager*")
			 (GetProcessorCount ((cores uint&)
					     (logical uint&))
					    :return "static void")
			 (AddJob2 ((a_Job Job*)))
			 (GetNumThreads ()
					:return "unsigned int"
					:code (return m_NumThreads))
			 (RunJobs ())
			 (ThreadDone ((n unsigned int)))
			 (MaxConcurrent ()
					:code (return m_NumThreads))
			 ))
		 "protected:"
		 "friend class JobThread;"
		 ,@(defmethods
		       :defs
		       `((GetNextJob ()
				     :return Job*)
			 (FindNextJob ()
				      :return Job*)))
		 "static JobManager* m_JobManager;"
		 "Job* m_JobList[256];"
		 "CRITICAL_SECTION m_CS;"
		 "HANDLE m_ThreadDone[64];"
		 "unsigned int m_NumThreads, m_JobCount;"
		 "JobThread* m_JobThreadList;"
		 ))))
    (define-module
	`(random ()
		 (do0
		  (split-header-and-code
		   (do0 (comments "header")
			)
		   (do0
		    (comments "implementation")
		    (do0 "static uint seed = 0x12345678;")
		    (do0
		     "static int numX=512,numY=512,numOctaves=7,primeIndex=0;"
		     "static float persistence=.5f;"
		     (let
			 ((primes
			    (curly
			     ,@(loop for e in
				     `(( 995615039 600173719 701464987 ) ( 831731269 162318869 136250887 )
				       ( 174329291 946737083 245679977 ) ( 362489573 795918041 350777237 )
				       ( 457025711 880830799 909678923 ) ( 787070341 177340217 593320781 )
				       ( 405493717 291031019 391950901 ) ( 458904767 676625681 424452397 )
				       ( 531736441 939683957 810651871 ) ( 997169939 842027887 423882827 ))
				     collect
				     `(curly ,@e)))))
		       (declare (type (array "static int" 10 3) primes))))
		    ))

		  ,@(defuns
			:defs
			`((Noise ((i const int)
				  (x )
				  (y ))
				 :return "static float"
				 :code
				 (let ((n (+ x (* 57 y))))
				   (setf n (^ (<< n 13) n))
				   (let ((a (aref primes i 0))
					 (b (aref primes i 1))
					 (c (aref primes i 2))
					 (tt (& (+ (* n (+ (* n n a)
							   b))
						   c)
						#x7fffffff)))
				     (return (- 1s0
						(/ (static_cast<float> tt)
						   1073741824.0s0))))))
			  (SmoothedNoise ((i const int)
					  (x const int)
					  (y))
					 :return "static float"
					 :code
					 (let ((corners (/ (+ ,@(loop for (a b c) in `((i x-1 y-1)
										       (i x+1 y-1)
										       (i x-1 y+1)
										       (i x+1 y+1))
								      collect
								      `(Noise ,a ,b ,c)))
							   16))
					       (sides (/ (+ ,@(loop for (a b c) in `((i x-1 y)
										     (i x+1 y)
										     (i x y-1)
										     (i x y+1))
								    collect
								    `(Noise ,a ,b ,c)))
							 8))
					       (center (/ (Noise i x y)
							  4)))
					   (return (+ corners sides center))))
			  (Interpolate ((a const float)
					(b )
					(x ))
				       :return "static float"
				       :code (let ((ft (* x 3.1415927s0))
						   (f (* .5s0 (- 1s0 (cosf ft)))))
					       (return (+ (* a (- 1 f))
							  (* b f)))))
			  (InterpolatedNoise ((i const int)
					      (x const float)
					      (y))
					     :return "static float"
					     :code (let ((integer_X (static_cast<int> x)
								    )
							 (integer_Y (static_cast<int> y))
							 (fractional_X (- x integer_X))
							 (fractional_Y (- y integer_Y))
							 ,@(loop for (e f) in `((integer_X integer_Y)
										(integer_X+1 integer_Y)
										(integer_X integer_Y+1)
										(integer_X+1 integer_Y+1))
								 and i from 0
								 collect
								 `(,(format nil "v~a" i)
								   (SmoothedNoise i ,e ,f)))
							 (i1 (Interpolate v1 v2 fractional_X))
							 (i2 (Interpolate v3 v4 fractional_X)))
						     (return (Interpolate i1 i2 fractional_Y))))
			  (noise2D ((x const float)
				    (y ))
				   :return float
				   :code (let ((total 0s0)
					       (frequency (static_cast<float> (<< 2 numOctaves)))
					       (amplitude 1s0))
					   (dotimes (i numOctaves)
					     (/= frequency 2)
					     (*= amplitude persistance)
					     (incf total
						   (* amplitude
						      (InterpolatedNoise (% (+ primeIndex 1)
									    10)
									 (/ x frequency)
									 (/ y frequency)))))
					   (return (/ total frequency))))
			  ))

		  ,@(defuns
			:defs
			`((RandomUInt ()
				      :return uint
				      :code (do0
					     (^= seed (<< seed 13))
					     (^= seed (>> seed 17))
					     (^= seed (<< seed 5))
					     (return seed)))
			  (RandomUInt ((seed uint&))
				      :return uint
				      :code (do0
					     (^= seed (<< seed 13))
					     (^= seed (>> seed 17))
					     (^= seed (<< seed 5))
					     (return seed)))
			  (RandomFloat ()
				       :return float
				       :code
				       (return (* (RandomUInt)
						  2.3283064365387e-10f)))
			  (RandomFloat ((seed &uint))
				       :return float
				       :code
				       (return (* (RandomUInt seed)
						  2.3283064365387e-10f)))
			  (Rand ((range float))
				:return float
				:code
				(return (* (RandomFloat) range)))
			  
			  (noise2D ((x const float)
				    (y const float))
				   :return float))))))

    (define-module
	`(file_helper ()
		      (do0
		       (split-header-and-code
			(do0 (comments "header")
			     )
			(do0
			 (comments "implementation")
		     
		     
			 ))
		       ,@(defuns
			     :defs
			     `((FileExists  ((f const char*))
					    :return bool
					    :code (let (((s f)))
						    (declare (type ifstream (s f)))
						    (return (s.good))))
			       (RemoveFile  ((f const char*))
					    :return bool
					    :code (if (FileExists f)
						      (return (not (remove f)))
						      (return false)))
			       (FileSize  ((f string))
					  :return uint
					  :code (let (((s f)))
						  (declare (type ifstream (s f)))
						  ;; FIXME: is this really returning file size?
						  (return (s.good))))
			       (TextFileRead ((f const char*))
					     :return string
					     :code
					     (let (((s f)))
					       (declare (type ifstream (s f)))
					       ;; FIXME: is this really returning file size?
					       (return (s.good))))
			       ))
		       )))
    (define-module
	`(math ()
	       (do0
		(split-header-and-code
			(do0 (comments "header")
			     )
			(do0
			 (comments "implementation")
			 
		     
			 ))
		,@(defuns
			     :defs
		      `((make_float3 ((x float)
				      (y float)
				      (z float))
				     :return float3
				     :code (let ((f))
					     (declare (type float3 f))
					     ,@(loop for e in `(x y z) collect
						     `(setf (dot f ,e)
							    ,e))
					     (return f)))
			(dot ((a float3)
			      (b float3))
			     :return float
			     :code (return (+ ,@(loop for e in `(x y z)
						      collect
						      `(* (dot a ,e)
							  (dot b ,e)))))
			     )
			(rsqrtf ((f float))
				:return float
				:code (return (/ 1s0
						 (sqrtf f))))
			(operator* ((a float3)
				    (s float ))
				   :return float3
				   :code
				   (return (make_float3 ,@(loop for e in `(x y z)
								collect
								`(* s (dot a ,e))))))
			(normalize ((v float3))
				   :return float3
				   :code
				   (let ((invLen (rsqrtf ("dot" v v))))
				     (return (* v invLen))))
			(cross ((a float3)
				(b float3))
			       :return float3
			       :code
			       (return (make_float3
					,@(loop for (e f) in `((y z)
							       (z x)
							       (x y))
						collect
						`(- (* (dot a ,e)
						       (dot b ,f))
						    (* (dot a ,f)
						       (dot b ,e)))))))
			      ))
		)))
    (define-module
	`(cl_buffer ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")
			   ;; needs cl_kernel
			   ))
		   
		     (defclass Buffer ()
		       "public:"
		       (space enum
			(curly
			 ,@(loop for (e f) in `((DEFAULT 0)
						(TEXTURE 8)
						(TARGET 16)
						(READONLY 1)
						(WRITEONLY 2))
				 collect
				 `(= ,e ,f))))
		       ,@(defmethods
			     :defs
			     `((Buffer  ()
					:decl ((construct (hostBuffer 0)))
					:return :constructor
					:code
					(do0
					 (comments "// nothing" ))
					#+nil (do0
					 (unless Kernel--clinitialized
					   (unless Kernel--InitCL
					     ,(logprint "failed to initialize opencl"))
					   (setf Kernel--clinitialized true))
					 (let ((type tt)
					       (ownData false)
					       (rwFlags CL_MEM_READ_WRITE))
					   (when (& tt READONLY)
					     (setf rwFlags CL_MEM_READ_ONLY))
					   (when (& tt WRITEONLY)
					     (setf rwFlags CL_MEM_WRITE_ONLY))
					   (if (== 0 (& tt (logior TEXTURE TARGET)))
					       (do0
						(setf size N
						      textureID 0 ;; not representing a texture
						      deviceBuffer (clCreateBuffer
								    (Kernel--GetContext)
								    rwFlags
								    (* size 4)
								    0 0)
						      hostBuffer (static_cast<uint*> ptr)
						      ))
					       (do0
						(setf textureID N)
						(unless Kernel--candoInterop
						  ,(logprint "didn't expect to get here"))
						(let ((err 0))
						  (if (== TARGET tt)
						      (setf deviceBuffer
							    (clCreateFromGLTexture
							     (Kernel--GetContext)
							     CL_MEM_WRITE_ONLY
							     GL_TEXTURE_2D
							     0
							     N
							     &error))
						      (setf deviceBuffer
							    (clCreateFromGLTexture
							     (Kernel--GetContext)
							     CL_MEM_READ_ONLY
							     GL_TEXTURE_2D
							     0
							     N
							     &error)))
						  (CHECKCL err)
						  (setf hostBuffer 0))
						))))
				      
					)
			       (Buffer  ((N unsigned int)
					 (tt unsigned int :default DEFAULT)
					 (ptr void* :default 0))
					
					:return :constructor
					:code
					(do0
					 (setf type tt
					       ownData false)
					 (let ((rwFlags CL_MEM_READ_WRITE))
					   (when (& tt READONLY)
					     (setf rwFlags CL_MEM_READ_ONLY))
					   (when (& tt WRITEONLY)
					     (setf rwFlags CL_MEM_WRITE_ONLY))
					   (if (== 0 (& tt (logior TEXTURE TARGET)))
					       (do0
						(setf size N
						      textureID 0
						      deviceBuffer (clCreateBuffer
								    (Kernel--GetContext)
								    rwFlags (* size 4) 0 0)
						      hostBuffer (static_cast<uint*> ptr))
						)
					       (do0
						(setf textureID N)
						(unless Kernel--candoInterop
						  ,(logprint "didn't expect to get here"))
						(let ((err 0))
						  (if (== TARGET tt)
						      (do0
						       (setf deviceBuffer (clCreateFromGLTexture
									   (Kernel--GetContext)
									   CL_MEM_WRITE_ONLY
									   GL_TEXTURE_2D
									   0 N &err))
						       )
						      (do0
						       (setf deviceBuffer (clCreateFromGLTexture
									   (Kernel--GetContext)
									   CL_MEM_READ_ONLY
									   GL_TEXTURE_2D
									   0 N &err))
						       (CHECKCL err)
						       (setf hostBuffer 0)
						       )))
						))))
				      
					)
			       (~Buffer  ()
					
					:return :constructor
					:code
					(when ownData
					  (delete hostBuffer))
				      
					)
			       (GetDevicePtr ()
					     :return cl_mem*
					     :code (return &deviceBuffer))
			       (GetHostPtr ()
					   :return "unsigned int*"
					   :code (return &hostBuffer))
			       (CopyToDevice ((blocking bool :default true))
					     :code
					     (let ((err (static_cast<cl_int> 0)))
					       (CHECKCL (= err
							   (clEnqueueWriteBuffer
							    (Kernel--GetQueue)
							    deviceBuffer
							    blocking
							    0
							    (* size 4)
							    hostBuffer
							    0 0 0)))))
			       (CopyToDevice2 ((blocking bool)
					       (e cl_event* :default 0)
					       (s const size_t :default 0))
					      :code
					      (do0
					       (comments "this uses the second queue")
					       (let ((err (static_cast<cl_int> 0)))
					       (CHECKCL (= err
							   (clEnqueueWriteBuffer
							    (Kernel--GetQueue2)
							    deviceBuffer
							    blocking
							    0
							    (? (== 0 s)
							       (* size 4)
							       (* s 4))
							    hostBuffer
							    0 0
							    eventToSet))))
					       )
					      
					      )
			       (CopyFromDevice ((blocking bool :default true))
					       :code
					       (let ((err (static_cast<cl_int> 0)))
					       (CHECKCL (= err
							   (clEnqueueReadBuffer
							    (Kernel--GetQueue)
							    deviceBuffer
							    blocking
							    0
							    (* size 4)
							    hostBuffer
							    0 0 0)))))
			       (CopyTo ((buffer Buffer*))
				       :code
				       (clEnqueueCopyBuffer (Kernel--GetQueue)
							    deviceBuffer
							    buffer->deviceBuffer
							    0 0
							    (* size 4)
							    0 0 0))
			       (Clear ()
				      :code
				      (let ((value (static_cast<uint> 0))
					    (err (static_cast<cl_int> 0)))
					(CHECKCL (= err
						    (clEnqueueFillBuffer
						     (Kernel--GetQueue)
						     deviceBuffer
						     &value
						     4 0 (* size 4)
						     0 0 0)))))
			       ))
		       "unsigned int* hostBuffer;"
		       "cl_mem deviceBuffer, pinnedBuffer;"
		       "unsigned int type, size, textureID;"
		       "bool ownData;"
		       ))))

    (define-module
	`(cl_kernel ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")
			   ;; needs textfileread helper
			   ))
		   
		     (defclass Kernel ()
		       "friend class Buffer;"
		       "public:"
		       
		       ,@(defmethods
			     :defs
			     `((Kernel  ((file char*)
					 (entryPoint char*))
					:return :constructor
					:code
					(let ((size (static_cast<size_t> 0)
						    )
					      (pos size)
					      (err (static_cast<cl_int> 0))
					      (csText (TextFileRead file))
					      (incLines 0))
					  (unless (csText.size)
					    ,(logprint "file not found" `(file)))
					  (while true
						 (do0
						  (setf pos (csText.find (string "#include")))
						  (when (== pos string--npos)
						    break))
						 (let ((tmp))
						   (declare (type string tmp))
						   (when (< 0 pos)
						     (setf tmp (csText.substr 0 (- pos 1))))
						   (do0
						    (setf pos (csText.find (string "\\\""))
							  )
						    (when (== pos string--npos)
						      ,(logprint "expected double quote after #include in shader"))
						    
						    )
						   (let ((end (csText.find (string "\\\"")
									   (+ pos 1))))
						     (when (== end string--npos)
						       ,(logprint "expected second double quote after #include in shader")))
						   (let ((file (csText.substr (+ pos 1)
									      (- end pos 1)))
							 (incText (TextFileRead (file.c_str)))
							 (p (incText.c_str)))
						     (while p
							    (incf incLines)
							    (setf p (strstr (+ p 1)
									    (string "\\n"))))
						     (decf incLines 2)
						     (incf tmp incText)
						     (incf tmp (csText.substr (+ end 1)
									      string--npos))
						     (setf csText tmp))))
					  (let ((source (csText.c_str))
						(size (strlen source))
						(program (clCreateProgramWithSource
							  context
							  1
							  ("static_cast<const char**>" &source)
							  &size
							  &err)))
					    (CHECKCL err)
					    (setf err (clBuildProgram
						       program
						       0
						       nullptr
						       (string "-cl-fast-relaxed-math -cl-mad-enable  -cl-denorms-are-zero -cl-no-signed-zeros -cl-unsafe-math-optimizations")
						       nullptr
						       nullptr))
					    (unless (== CL_SUCCESS err)
					      (unless log
						(setf log (new (aref char (* 256 1024)))))
					      (setf (aref log 0) 0)
					      (clGetProgramBuildInfo program
								     (getFirstDevice context)
								     CL_PROGRAM_BUILD_LOG
								     (* 100 1024)
								     log
								     nullptr)
					      (setf (aref log 2048) 0)
					      ,(logprint "build error" `(log)))
					    (setf kernel (clCreateKernel program
									 entryPoint
									 &err))
					    )
					 )
					
				      
					)
			       (Kernel  ((existingProgram cl_program&)
					 (entryPoint char*))
					:return :constructor
					:code
					(let ((err (static_cast<cl_int>)))
					  (setf program existingProgram
						kernel (clCreateKernel program entryPoint &err))
					  (do0 (unless kernel
						   ,(logprint "create kernel failed: entry point not found."))
						 (CHECKCL err)))
					
				      
					)
			       (~Kernel  ()
					:return :constructor
					:code
					(do0
					 (when kernel
					   (clReleaseKernel kernel)
					   )
					 (when program
					   (clReleaseProgram program)))
					
				      
					)
			       (GetKernel ()
					  :return cl_kernel&
					  :code
					  (return kernel))
			       (GetProgram ()
					  :return cl_program&
					  :code
					  (return program))
			       (GetQueue ()
					  :return "static cl_command_queue&"
					  :code
					  (return queue))
			       (GetQueue2 ()
					  :return "static cl_command_queue&"
					  :code
					  (return queue2))
			       (GetContext ()
					  :return "static cl_context&"
					  :code
					  (return context))
			       (GetDevice ()
					  :return "static cl_device_id&"
					  :code
					  (return device))
			       (Run ((eventToWaitFor cl_event* :default 0)
				     (eventToSet cl_event* :default 0))
				    :code
				    (do0
				     (glFinish)
				     (let ((err (clEnqueueNDRangeKernel
						 queue
						 kernel
						 2 0
						 workSize
						 localSize
						 (? eventToWaitFor 1 0)
						 eventToWaitFor
						 eventToSet)))
				       (CHECKCL err)
				       (clFinish queue))
				     ))
			       (Run ((buffers cl_mem*)
				     (count int :default 1)
				     (eventToWaitFor cl_event* :default 0)
				     (eventToSet cl_event* :default 0)
				     (acq cl_event* :default 0)
				     (rel cl_event* :default 0))
				    :code
				    (do0
				     (let ((err (static_cast<cl_int> 0)))
				      (if Kernel--candoInterop
					  (do0
					   (CHECKCL (= err (clEnqueueAcquireGLObjects
							    queue count buffers 0 0 acq)))
					   (CHECKCL (= err (clEnqueueNDRangeKernel
							    queue
							    kernel
							    2 0
							    workSize
							    localSize
							    (? eventToWaitFor 1 0)
							    eventToWaitFor
							    eventToSet)))
					   (CHECKCL (= err (clEnqueuReleaseGLObjects queue
										     count
										     buffers
										     0 0 rel)))
					   )
					  (do0
					   (CHECKCL (= err (clEnqueueNDRangeKernel
							    queue
							    kernel
							    2 0
							    workSize
							    localSize
							    (? eventToWaitFor 1 0)
							    eventToWaitFor
							    eventToSet)))
					   )))
				     
				     ))
			       (Run ((buffer Buffer*)
				     (tileSize const int2)
				     (eventToWaitFor cl_event* :default 0)
				     (eventToSet cl_event* :default 0)
				     (acq cl_event* :default 0)
				     (rel cl_event* :default 0)
				     )
				    :code
				    (do0
				     (unless arg0set
				       ,(logprint "kernel expects at least one argument, none set."))
				     (let ((err (static_cast<cl_int> 0)))
				      (if Kernel--candoInterop
					  (let ((localSize (curly (static_cast<size_t> tileSize.x)
								  (static_cast<size_t> tileSize.y)))
						(count 1))
					    (declare (type (array size_t 2) localSize)) (do0
						(CHECKCL (= err (clEnqueueAcquireGLObjects
								 queue count (buffer->GetDevicePtr) 0 0 acq)))
						(CHECKCL (= err (clEnqueueNDRangeKernel
								 queue
								 kernel
								 2 0
								 workSize
								 localSize
								 (? eventToWaitFor 1 0)
								 eventToWaitFor
								 eventToSet)))
						(CHECKCL (= err (clEnqueuReleaseGLObjects queue
											  count
											  buffers
											  0 0 rel)))
						))
					  (do0
					   (CHECKCL (= err (clEnqueueNDRangeKernel
							    queue
							    kernel
							    2 0
							    workSize
							    localSize
							    (? eventToWaitFor 1 0)
							    eventToWaitFor
							    eventToSet)))
					   )))
				     ))
			       (Run ((count const size_t)
				     (localSize const int2 :default (make_int2 32 2))
				     (eventToWaitFor cl_event* :default 0)
				     (eventToSet cl_event* :default 0)
				     )
				    :code
				    (do0
				     
				     (let ((err (static_cast<cl_int> 0)))
				       (CHECKCL (= err (clEnqueueNDRangeKernel
								 queue
								 kernel
								 1 0
								 &count
								 (? (== 0 localSize)
								    0 &localSize)
								 (? eventToWaitFor 1 0)
								 eventToWaitFor
								 eventToSet))))
				     ))
			       (Run2D ((count const int2)
				       (lsize const int2)
				     (eventToWaitFor cl_event* :default 0)
				     (eventToSet cl_event* :default 0)

				       )
				    :code
				    (do0
				     (let ((localSize (curly (static_cast<size_t> lsize.x)
							     (static_cast<size_t> lsize.y)))
					   (workSize (curly (static_cast<size_t> count.x)
							    (static_cast<size_t> count.y)))
					   (err (static_cast<cl_int> 0)))
				      (CHECKCL (= err (clEnqueueNDRangeKernel
						       queue
						       kernel
						       2 0
						       workSize
						       localSize
						       (? eventToWaitFor 1 0)
						       eventToWaitFor
						       eventToSet))))
				     ))
			       ,@(loop for e in `((buffer cl_mem* :size cl_mem :arg buffer)
						  (buffer Buffer* :size cl_mem :arg (buffer->GetDevicePtr))
						  (value float )
						  (value int)
						  (value float2)
						  (value float3)
						  (value float4))
				       collect
				       (destructuring-bind (name type &key (size type) (code) (arg "&value")) e
					`(SetArgument ((idx int)
						       (,name ,type))
						      :code
						      (do0
						       (clSetKernelArg
							kernel idx
							(sizeof ,size)
							,arg)
						       (setf arg0set
							     (logior
							      arg0set
							      (== 0 idx))))
						      )))
			       (InitCL ()
				       :return bool
				       :code (do0
					(do0
					 "cl_platform_id platform;"
					 #+nil(do0 "cl_device_id* devices;"
						   "cl_uint devCount;"
						   "cl_int err;"))
					(let ((err (static_cast<cl_int> 0))
					      (devCount (static_cast<cl_uint> 0))
					      (devices (static_cast<cl_device_id*> nullptr)))
					  (unless (CHECKCL (= err (getPlatformID &platform)))
					    (return false))
					  (unless (CHECKCL (= err (clGetDeviceIDs platform
										  CL_DEVICE_TYPE_ALL
										  0
										  nullptr
										  &devCount)))
					    (return false)
					    )
					  (setf devices (new (aref cl_device_id devCount)))
					  (unless (CHECKCL (= err (clGetDeviceIDs platform CL_DEVICE_TYPE_ALL
										  devCount devices nullptr)))
					    (return false))
					  ;; deviceUsed as -1 seems to be a weird hack to identify device 0
					  ;; why not use int here?
					  (let ((deviceUsed (static_cast<uint> -1))
						(endDev (static_cast<uint> (- devCount 1))))
					    (dotimes (i endDev)
					      (let ((extensionSize (static_cast<size_t> 0)))
						(CHECKCL (= err (clGetDeviceInfo
								 (aref devices i)
								 CL_DEVICE_EXTENSIONS
								 0
								 nullptr
								 &extensionSize)))
						(when (< 0 extensionSize)
						  (let ((extensions (static_cast<char*> (malloc exensionSize))))
						    (CHECKCL (= err (clGetDeviceInfo
								     (aref devices i)
								     CL_DEVICE_EXTENSIONS
								     extensionSize
								     extensions
								     &extensionSize)))
						    "string devices( extensionsions );"
						    (free extensinos))
						  (let ((o (static_cast<size_t> 0))
							(s (devices.find (char " ")
									 o)))
						    (while (!= s devices.npos)
							   (let ((subs (devices.substr o (- s o))))
							     ;; check if device can do gl/cl interop
							     (unless (strcmp (string "cl_khr_gl_sharing")
									     (subs.c_str))
							       (setf deviceUsed 1)
							       break)
							     (space do
								    (progn
								      (setf o (+ s 1)
									    s (devices.find (char " ")
											    o)))
								    while
								    (paren (== s o)))))
						    (when (< -1 deviceUsed)
						      break)))))
					    (let ((props
						    (curly
						     ,@(loop for (e f) in
							     `((CL_CONTEXT_PLATFORM platform)
							       ;; this looks like windows stuff
					; (CL_WGL_HDC_KHR (wglGetCurrentDC))
							       ;; fixme what to use in linux?
							       (CL_GL_CONTEXT_KHR (glfwGetWGLContext window)))
							     appending
							     `(,e
							       (static_cast<cl_context_properties> ,f))
							     )
						     0)))
					      (declare (type (array cl_context_properties "") props))
					      ;; attempt to create context with requested features
					      (setf candoInterop true
						    context (clCreateContext props
									     1
									     (ref (aref devices
											deviceUsed))
									     nullptr
									     nullptr
									     &err))
					      (when (!= 0 err)
						,(logprint "no capable opencl device found."))
					      (let ((device (getFirstDevice context)))
						;; fixme: is this err set?
						(unless (CHECKCL err)
						  (return false))
						(let ((device_string)
						      (device_platform))
						  (declare (type (array char 1024) device_string
								 device_platform))
						  ,@(loop for (e f) in `((NAME device_string)
									 (VERSION device_platform)) collect
							  `(clGetDeviceInfo (aref devices deviceUsed)
									    ,(format nil "CL_DEVICE_~a" e)
									    1024
									    (ref ,f)
									    nullptr))
						  ,(logprint "device" `(deviceUsed
									device_string
									device_platform))
						  ;; second queue is for async copies
						  ,@(loop for e in `(queue queue2)
							  collect
							  `(do0
							    (setf ,e
								  (clCreateCommandQueue
								   context
								   (aref devices deviceUsed)
								   0 ;; profiling
								   &err))
							    (unless (CHECKCL err)
							      (return false))))
						  (delete devices)
						  (return true)
						  ))
					      )
					   					    )))
				       )
			       
			       
			       ))
		       "private:"
		       "cl_kernel kernel;"
		       "cl_mem vbo_cl;"
		       "cl_program program;"
		       "bool arg0set = false;"
		       "inline static cl_device_id device;"
		       "inline static cl_context context;" ;; limits us to one device
		       "inline static cl_command_queue queue, queue2;"
		       "inline static char* log = 0;"
		       "public:"
		       "inline static bool candoInterop = false;"
		       ))))

    (define-module
	`(cl_helper ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   "#define CHECKCL(r) CheckCL(r, __FILE__, __LINE__ )"
			   )
		      (do0 (comments "implementation")
			   
			   ))
		     (defun CheckCL (result file line)
		       (declare (type cl_int result)
				(type "const char*" file)
				(type int line)
				(values bool))
		       (case result
			 (CL_SUCCESS (return true))
			 ,@(loop for e in `(DEVICE_NOT_FOUND DEVICE_NOT_AVAILABLE COMPILER_NOT_AVAILABLE MEM_OBJECT_ALLOCATION_FAILURE OUT_OF_RESOURCES OUT_OF_HOST_MEMORY PROFILING_INFO_NOT_AVAILABLE MEM_COPY_OVERLAP IMAGE_FORMAT_MISMATCH IMAGE_FORMAT_NOT_SUPPORTED BUILD_PROGRAM_FAILURE MAP_FAILURE MISALIGNED_SUB_BUFFER_OFFSET EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST INVALID_VALUE INVALID_DEVICE_TYPE INVALID_PLATFORM INVALID_DEVICE INVALID_CONTEXT INVALID_QUEUE_PROPERTIES INVALID_COMMAND_QUEUE INVALID_HOST_PTR INVALID_MEM_OBJECT INVALID_IMAGE_FORMAT_DESCRIPTOR INVALID_IMAGE_SIZE INVALID_SAMPLER INVALID_BINARY INVALID_BUILD_OPTIONS INVALID_PROGRAM INVALID_PROGRAM_EXECUTABLE INVALID_KERNEL_NAME INVALID_KERNEL_DEFINITION INVALID_KERNEL INVALID_ARG_INDEX INVALID_ARG_VALUE INVALID_ARG_SIZE INVALID_KERNEL_ARGS INVALID_WORK_DIMENSION INVALID_WORK_GROUP_SIZE INVALID_WORK_ITEM_SIZE INVALID_GLOBAL_OFFSET INVALID_EVENT_WAIT_LIST INVALID_EVENT INVALID_OPERATION INVALID_GL_OBJECT INVALID_BUFFER_SIZE INVALID_MIP_LEVEL INVALID_GLOBAL_WORK_SIZE)
				 collect
				 `(,(format nil "CL_~a" e)
				   ,(logprint (format nil "CL error: ~a" e)
					      `(file line))))
			
			 )
		       (return false))
		     (defun getFirstDevice (context)
		       (declare (type cl_context context)
				(values  cl_device_id))
		       (let ((dataSize (static_cast<size_t> 0))
			     (devices (static_cast<cl_device_id*> nullptr))
			     )
			 (clGetContextInfo context CL_CONTEXT_DEVICES 0 nullptr &dataSize)
			 (setf devices (static_cast<cl_device_id*> (malloc dataSize)))
			 (clGetContextInfo context CL_CONTEXT_DEVICES dataSize devices nullptr)
			 (let ((first (aref devices 0)))
			   (free devices)
			   (return first))))
		     (defun getPlatformID (platform)
		       (declare (type cl_platform_id* platform)
				(values  cl_int))
		       (let ((chBuffer)
			     (num_platforms (static_cast<cl_uint> 0))
			     (devCount  (static_cast<cl_uint> 0))
			     (ids (static_cast<cl_platform_id*> nullptr))
			     (err (static_cast<cl_int> 0)))
			 (declare (type (array char 1024) chBuffer))

			 (CHECKCL (= err (clGetPlatformIDs 0 nullptr &num_platforms)))
			 (when (== 0 num_platforms)
			   ,(logprint "no platforms"))
			 (setf ids (static_cast<cl_platform_id*> (malloc (* num_platforms (sizeof cl_platform_id)))))
			 (do0
			  (setf err (clGetPlatformIDs num_platforms ids nullptr)
				)
			  (let ((deviceType (curly CL_DEVICE_TYPE_GPU
						   CL_DEVICE_TYPE_CPU))
				(deviceOrder (curly (curly (string "NVIDIA")
							   (string "AMD")
							   (string ""))
						    (curly (string "")
							   (string "")
							   (string "")))))
			    (declare (type (array cl_uint 2) deviceType)
				     (type (array char* 2 3) deviceOrder))
			    (dotimes (i num_platforms)
			      (CHECKCL (= err (clGetPlatformInfo
					       (aref ids i)
					       CL_PLATFORM_NAME
					       1024
					       &chBuffer
					       nullptr)))
			      ,(logprint "opencl platform" `(i chBuffer)))
			    (dotimes (j 2)
			      (dotimes (k 3)
				(dotimes (i num_platforms)
				  (setf err (clGetDeviceIDs (aref ids i)
							    (aref deviceType j)
							    0 nullptr
							    &devCount))
				  (when (or (!= CL_SUCCESS err)
					    (== 0 devCount))
				    continue)
				  (CHECKCL (= err (clGetPlatformInfo (aref ids i)
								     CL_PLATFORM_NAME
								     1024
								     &chBuffer
								     nullptr)))
				  (when (aref deviceOrder j k 0)
				    (unless (strstr chBuffer
						    (aref deviceOrder j k))
				      continue))
				  ,(logprint "opencl device" `(chBuffer))
				  (setf *platform (aref ids i)
					j 2
					k 3)
				  break)))
			    ))
			 (free ids)
			 (return CL_SUCCESS)))
		     )))

    (define-module
	`(scene_texture ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")
			   
			   ))
		   
		     (defclass Texture ()
		       "public:"
		       ,@(defmethods
			  :defs
			  `((Texture ()
				     :return :constructor
				     :decl ((construct (m_B32 0)))
				     )
			    (Init ((a_File char*))
				  :code
				  (do0
				   (setf m_B32 0)
				   (let ((f (fopen a_File (string "rb"))))
				     (unless f
				       (return))
				     (let ((fif FIF_UNKNOWN))
				       (setf fif (FreeImage_GetFileType a_File 0))
				       (when (== FIF_UNKNOWN fif)
					 (setf fif (FreeImage_GetFIFFromFilename a_File)))
				       (let ((tmp (FreeImage_Load fif a_File))
					     (dib (FreeImage_ConvertTo32Bits tmp)))
					 (FreeImage_Unload tmp)
					 (let ((bits (FreeImage_GetBits dib)))
					   (setf m_Width (FreeImage_GetWidth dib)
						 m_Height (FreeImage_GetHeight dib)
						 m_B32 (static_cast<uint*> (MALLOC64 (* m_Width
											m_Height
											(sizeof uint)))))
					   (dotimes (y m_Height)
					     (let ((line (FreeImage_GetScanLine dib (- m_Height 1 y))))
					       (memcpy (+ m_B32
							  (* y m_Width))
						       line
						       (* m_Width (sizeof uint)))))
					   (FreeImage_Unload dib)))))))
			    (GetBitmap ()
				       :decl ((const))
				       :return "const unsigned int*"
				       :code
				       (do0
					(return m_B32)))
			    (GetWidth ()
				       :decl ((const))
				       :return "const unsigned int"
				       :code
				       (do0
					(return m_Width)))
			    (GetHeight ()
				       :decl ((const))
				       :return "const unsigned int"
				       :code
				       (do0
					(return m_Height)))))
		       "unsigned int* m_B32;"
		       "unsigned int m_Width, m_Height;"
		       ))))

    (define-module
	`(scene_material ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")
			   
			   ))
		   
		     (defclass Material ()
		       "public:"
		       ,@(defmethods
			  :defs
			  `((Material ()
				      :return :constructor
				      :code (do0
					     (setf m_Texture 0
						   m_Name (new (aref char 64)))
					     (SetDiffuse
					      (make_float3 .8s0 .8s0 .8s0))
					     )
				     )
			    (~Material ()
				       :return :constructor
				       :code (delete m_Name)
				     )
			    (SetDiffuse ((a_Diff const float3&))
				  :code
				  (do0
				   (setf m_Diff a_Diff)))
			    (SetTexture ((a_Texture const Texture*))
				  :code
				  (do0
				   (setf m_Texture (static_cast<Texture*> a_Texture))))
			    (SetName ((a_Name char*))
				  :code
				  (do0
				   (strcpy m_Name a_Name)))
			    (SetIdx ((a_Idx uint))
				  :code
				  (do0
				   (setf m_Idx a_Idx)))
			    (GetDiffuse ()
					:return "const float3"
					:decl ((const))
					:code (return m_Diff))
			    (GetTexture ()
					:return "const Texture*"
					:decl ((const))
					:code (return m_Texture))
			    (GetName ()
					:return "char*"
					:code (return m_Name))
			    (GetIdx ()
					:return uint
					:code (return m_Idx))
			    ))
		       "private:"
		       "Texture* m_Texture;"
		       "uint m_Idx;"
		       "float3 m_Diff;"
		       "char* m_Name;"))))
    (define-module
	`(scene_matmanager ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")
			   ;; uses static Scene* scene
			   ))
		   
		     (defclass MatManager ()
		       "public:"
		       ,@(defmethods
			  :defs
			  `((MatManager ()
				      :return :constructor
				      :code (do0
					     (setf m_Mat (new (aref Material* 1024)))
					     (setf (aref m_Mat 0) (new (Material)))
					     (-> (aref m_Mat 0)
						 (SetName (string "DEFAULT")))
					     (setf m_NrMat 1)
					     )
				     )
			    (LoadMTL ((a_File char*))
				     :code
				     (do0
				      (let ((f (fopen a_File (string "r"))))
					(unless f
					  return)
					(let ((curmat (static_cast<uint> 0))
					      (buffer)
					      (cmd))
					  (declare (type (array char 256) buffer)
						   (type (array char 128) cmd))
					  (while (not (feof f))
						 (fgets buffer 250 f)
						 (sscanf buffer (string "%s")
							 cmd)
						 (unless (_stricmp cmd (string "newmtl"))
						   (incf m_NrMat)
						   (setf curmat m_NrMat
							 (aref m_Mat curmat) (new (Material)))
						   (let ((matname))
						     (declare (type (array char 128) matname))
						     (sscanf (+ buffer
								(strlen cmd))
							     (string "%s")
							     matname)
						     (-> (aref m_Mat curmat)
							 (SetName matname)))
						   )
						 (unless (_stricmp cmd (string "Kd"))
						   (let ((r 0s0)
							 (g 0s0)
							 (b 0s0))
						     (sscanf (+ buffer 3)
							     (string "%f %f %f"
								     )
							     &r &g &b)
						     (let ((c (make_float3 r g b)))
						       (-> (aref m_Mat curmat)
							   (SetDiffuse c))))
						   )
						 (unless (_stricmp cmd (string "map_Kd"))
						   (let ((tname)
							 (fname))
						     (declare (type (array char 128) tname)
							      (type (array char 128) fname))
						     (setf (aref tname 0) 0
							   (aref fname 0) 0)
						     (sscanf (+ buffer 7)
							     (string "%s")
							     tname)
						     (when (aref tname 0)
						       ;; FIXME: how to get static Scene* scene here?
						       (strcpy fname scene->path)
						       (unless (strstr tname (string "textures/"))
							 (strcat fname (string "textures/")))
						       (strcat fname tname)
						       (let ((tex (new Texture)))
							 (tex->Init fname)
							 (unless (tex->m_B32)
							   (strcpy fname scene->path)
							   (strcat fname tname)
							   (tex->Init fname))
							 (-> (aref m_Mat curmat)
							     (SetTexture tex))))))
						 )
					  ))))
			    (FindMaterial ((a_Name char*))
					  :return Material*
					  :code
					  (do0
					   (dotimes (i m_NrMat)
					     (unless (_stricmp (-> (aref m_Mat i)
								   (GetName))
							       a_Name)
					       (return (aref m_Mat i))))
					   (return (aref m_Mat 0))
					   ))
			    (GetMaterial ((a_Idx int))
					  :return Material*
					  :code
					  (do0
					   (return (aref m_Mat a_Idx))))
			    (Reset ()
				   :code
				   (setf m_NrMat 0))
			    (AddMaterial ((a_Mat Material*))
					 :code
					 (do0
					  (incf m_NrMat)
					  (setf (aref m_Mat m_NrMat)
						a_Mat)))
			    (GetMatCount ()
					 :return uint
					 :code (return m_NrMat))))
		       "private:"
		       "Material** m_Mat;"
		       "uint m_NrMat;"))))
    
    (define-module
	`(scene_primitive ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   )
		      (do0 (comments "implementation")
			   ;; needs normalize and cross
			   ))

		     
		     
		     (defclass Primitive ()
		       "public:"
		       ,@(defmethods
			  :defs
			  `((Primitive ()
				      :return :constructor
				      )
			    (Init ((a_V1 Vertex*)
				   (a_V2 Vertex*)
				   (a_V3 Vertex*))
				  :code
				  (do0
				   ,@(loop for i below 3
					   collect
					   `(setf (aref m_Vertex ,i)
						  ,(format nil "a_V~a" (+ i 1))))
				   (UpdateNormal)))
			    (GetMaterial ()
					 :decl ((const))
					:return "const Material*"
					:code
					(return m_Material))
			    (SetMaterial ((a_Mat const Material*))
					 :code
					 (setf m_Material (static_cast<Material*> a_Mat)))
			    (SetNormal ((a_N const float3&))
				       :code
				       (setf m_N a_N))
			    (UpdateNormal ()
					  :code
					  (let ((c (normalize (- (-> (aref m_Vertex 1)
								     (GetPos))
								 (-> (aref m_Vertex 0)
								     (GetPos)))))
						(b (normalize (- (-> (aref m_Vertex 2)
								     (GetPos))
								 (-> (aref m_Vertex 0)
								     (GetPos))))))
					    (setf m_N (cross b c))))
			    (GetVertex ((a_Idx const uint))
				       :return "const Vertex*"
				       :decl ((const))
				       :code (return (aref m_Vertex a_Idx)))
			    (SetVertex ((a_Idx const uint)
					(a_Vertex Vertex*))
				       :code
				       (setf (aref m_Vertex a_Idx)
					     a_Vertex))
			    (GetNormal ((u float)
					(v float))
				       :decl ((const))
				       :return "const float3"
				       :code (return m_N))
			    ))
		       "float3 m_N;"
		       "Vertex* m_Vertex[3];"
		       "Material* m_Material;"))))

    (define-module
	`(scene ()
		    (do0
		     (split-header-and-code
		      (do0 (comments "header")
			   "static Scene* scene;")
		      (do0 (comments "implementation")
			   ))

		     
		     (defclass  aabb ()
		       "public:"
		       (defmethod aabb ()
			 (declare (values :constructor)
				  (construct ((bmin (make_float3 1000s0 1000s0 1000s0))
					      (bmax (make_float3 -1000s0 -1000s0 -1000s0))))))
		       "float3 bmin;"
		       "float3 bmax;")
		     
		     (defclass Scene ()
		       "public:"
		       ,@(defmethods
			  :defs
			  `((Scene ()
				      :return :constructor
				      :code (do0
					     (setf m_MatMan (new MatManager)))
				     )
			    (InitSceneState ()
					    :code
					    (setf m_Extends.bmin (make_float3 1000s0 1000s0 1000s0)
						  m_Extends.bmax (make_float3 -1000s0 -1000s0 -1000s0)
						  scene this))
			    (InitScene ((a_File const char*))
				       :return bool
				       :code
				       (do0
					(InitSceneState)
					(setf path (new (aref char 1024)))
					(setf file (new (aref char 1024)))
					(setf noext (new (aref char 1024)))
					(strcpy path a_File)
					(let ((pos path))
					  (do0 (while (strstr pos (string "/"))
						  (setf pos (+ (strstr pos (string "/"))
							       1)))
					       (while (strstr pos (string "\\\\"))
						      (setf pos (+ (strstr pos (string "\\\\"))
								   1))))
					  (setf *pos 0)
					  (setf pos (static_cast<char*> a_File))
					  (do0 (while (strstr pos (string "/"))
						  (setf pos (+ (strstr pos (string "/"))
							       1)))
					       (while (strstr pos (string "\\\\"))
						      (setf pos (+ (strstr pos (string "\\\\"))
								   1))))
					  (if (or (strstr a_File (string "/"))
						  (strstr a_File (string "\\\\")))
					      (strcpy file pos)
					      (strcpy file a_File))
					  (LoadOBJ a_File)
					  (strcpy noext file)
					  (setf pos noext)
					  (while (strstr pos (string "."))
						 ;; FIXME: why no +1 here?
						      (setf pos (strstr pos (string "."))
							    ))
					  (when (== *pos (string "."))
					    (setf *pos 0))
					  (return true))))
			    (GetExtends ()
					:return "const aabb&"
					:code
					(return m_Extends))
			    (SetExtends ((a_Box aabb))
					:code (setf m_Extends a_Box))
			    (GetMatManager ()
					   :return MatManager*
					   :code (return m_MatMan))
			    (UpdateSceneExtends
			     ()
			     :code
			     (let ((emin (make_float3 10000
						      10000
						      10000))
				   (emax (make_float3 -10000
						      -10000
						      -10000)))
			       (dotimes (j m_Primitives)
				 (dotimes (v 3)
				   (let ((pos (dot (aref m_Prim j)
						   (GetVertex v)
						   (->GetPos))))
				     (dotimes (a 3)
				       (do0 (when (< (aref pos.e a)
						     (emin.e a))
					      (setf (aref emin.e a)
						    (aref pos.e a)))
					    (when (< (emax.e a)
						     (aref pos.e a)
						     )
					      (setf (aref emax.e a)
						    (aref pos.e a))))))))
			       (setf m_Extends.bmin emin
				     m_Extends.bmax emax)))
			    (LoadOBJ
			     ((filename const char*))
			     :code
			     (let ((f (fopen filename (string "r"))))
			       (unless f
				 (return))
			       (let ((fcount (static_cast<uint> 0))
				     (ncount (static_cast<uint> 0))
				     (uvcount (static_cast<uint> 0))
				     (vcount (static_cast<uint> 0))
				     (buffer)
				     (cmd)
				     (objname))
				 (declare (type (array char 256) buffer cmd objname))
				 (while true
					(fgets buffer 250 f)
					(when (feof f)
					  break)
					(if (== (char "v")
						(aref buffer 0))
					    (do0
					     (case (aref buffer 1)
					       ((char " ") (incf vcount))
					       ((char "t") (incf uvcount))
					       ((char "n") (incf ncount))))
					    (do0
					     (when (and (== (char "f") (aref buffer 0))
							(== (char " ") (aref buffer 1)))
					       (incf fcount)))))
				 (fclose f)
				 (setf m_Prim (static_cast<Primitive*> (MALLOC64 (* fcount (sizeof Primitive)))))
				 (setf f (fopen filename (string "r")))
				 (let ((curmat (static_cast<Material*> nullptr))
				       (verts (static_cast<uint> 0))
				       (uvs (static_cast<uint> 0))
				       (normals (static_cast<uint> 0))
				       (vert (new (aref float3 vcount)))
				       (norm (new (aref float3 ncount)))
				       (tu (new (aref float uvcount)))
				       (tv (new (aref float uvcount)))
				       (vertex (static_cast<Vertex*> (MALLOC64 (* (sizeof Vertex)
										  (+ 4 (* fcount 3))))))
				       (vidx (static_cast<uint> 0))
				       (currobject))
				   (declare (type (array char 256) currobject))
				   (strcpy currobj (string "none"))
				   (setf m_Primitives 0)
				   (while true
					  (do0 (fgets buffer 250 f)
					       (when (feof f)
						 break))
					  (case (aref buffer 0)
					    ((char "v")
					     (case (aref buffer 1)
					       ((char " ")
						(let ((x 0s0)
						      (y 0s0)
						      (z 0s0))
						  (sscanf buffer (string "%s %f %f %f")
							  cmd &x &y &z)
						  (let ((pos (make_float3 x y z)))
						    (incf verts)
						    (setf (aref vert verts) pos)
						    (dotimes (a 3)
						      (do0 (when (< (aref pos.e a)
								    (m_Extends.bmin.e a))
							     (setf (aref m_Extends.bmin.e a)
								   (aref pos.e a)))
							   (when (< (m_Extends.bmax.e a)
								    (aref pos.e a))
							     (setf (aref m_Extends.bmax.e a)
								   (aref pos.e a))))))))
					       ((char "t")
						(let ((u 0s0)
						      (v 0s0))
						  (sscanf buffer (string "%s %f %f")
							  cmd &u &v)
						  (setf (aref tu uvs) u)
						  (incf uvs)
						  (setf (aref tv uvs) -v)))
					       ((char "n")
						(let ((x 0s0)
						      (y 0s0)
						      (z 0s0))
						  (sscanf buffer (string "%s %f %f %f")
							  cmd &x &y &z)
						  (incf normals)
						  (setf (aref norm normals)
							(make_float3 -x -y -z)))
						)))
					    (t break))
					  )
				   (fclose f)
				   (setf f (fopen filename (string "r")))
				   (while true
					  (do0 (fgets buffer 250 f)
					       (when (feof f)
						 break))
					  (case (aref buffer 0)
					    ((char "f")
					     ;; face
					     (let ((v)
						   (cu)
						   (cv)
						   (tex (curmat->GetTexture))
						   (vnr)
						   (vars (sscanf (+ buffer 2)
								 (string ,(format nil "~{~{~a~^/~}~^ ~}"
										  (loop for j below 3
											collect
											(loop for i below 3
											      collect "%i"))))
								 ,@(loop for i below 9
									 collect
									 `(ref (aref vnr ,i))))))
					       (declare (type (array Vertex* 3) v)
							(type (array float 3) cu cv)
							(type (array uint 9) vnr ))
					       (when (< vars 9)
						 (setf vars (sscanf (+ buffer 2)
								 (string ,(format nil "~{~{~a~^/~}~^ ~}"
										  (loop for j below 3
											collect
											(loop for i below 2
											      collect "%i"))))
								 ,@(loop for i in `(0 2 3 5 6 8)
									 collect
									 `(ref (aref vnr ,i)))))
						 (when (< vars 6)
						 (sscanf (+ buffer 2)
							 (string ,(format nil "~{~{~a~^//~}~^ ~}"
									  (loop for j below 3
										collect
										(loop for i below 2
										      collect "%i"))))
							 ,@(loop for i in `(0 2 3 5 6 8)
								 collect
								 `(ref (aref vnr ,i))))))
					       (dotimes (i 3)
						 (incf vidx)
						 (setf (aref v i)
						       (aref &vertex vidx))
						 (when tex
						   ;; FIXME: potential name clash vidx
						   (let ((vidx (- (aref vnr (+ (* i 3) 1))
								  1)))
						     (setf (aref cu i) (aref tu vidx)
							   (aref cv i) (aref tv vidx))))
						 (let ((nidx (- (aref vnr (+ 2 (* i 3)))
								1))
						       (vidx (- (aref vnr (* i 3))
								1)))
						   (-> (aref v i)
						       (SetNormal (aref norm nidx)))
						   (-> (aref v i)
						       (SetPos (aref vert vidx)))))
					       (incf m_Primitives)
					       (let ((p (ref (aref m_Prim m_Primitives))))
						 (when tex
						   ,@(loop for i below 3
							   collect
							   `(-> (aref v ,i)
								(SetUV
								 (* (aref cu ,i)
								    (tex->m_Width))
								 (* (aref cv ,i)
								    (tex->m_Height))))))
						 (p->Init (aref v 0)
							  (aref v 1)
							  (aref v 2))
						 (p->SetMaterial curmat)))
					     )
					    ((char "g")
					     (sscanf (+ buffer 2)
						     (string "%s")
						     objname)
					     (strcpy currobj objname))
					    ((char "m")
					     (unless (_strnicmp buffer (string "mtllib")
								6)
					       (let ((libname)
						     (fullname))
						 (declare (type (array char 128) libname)
							  (type (array char 256) fullname))
						 (sscanf (+ buffer 7)
							 (string "%s")
							 libname)
						 (strcpy fullname path)
						 (strcat fullname libname)
						 (m_MatMan->LoadMTL fullname))))
					    ((char "u")
					     (unless (_strnicmp buffer (string "usemtl")
								6)
					       (let ((matname)
						     )
						 (declare (type (array char 128) matname)
							 )
						 (sscanf (+ buffer 7)
							 (string "%s")
							 matname)
						 (setf curman (m_MatMan->FindMaterial matname))))))
					  )))))))
		       "uint m_Primitives;"
		       "uint m_MaxPrims;"
		       "aabb m_Extends;"
		       "MatManager* m_MatMan;"
		       "Primitive* m_Prim;"
		       "char *path;"
		       "char *file;"
		       "char *noext;"))))
    
    )
  
  (progn
    
    
    (progn				;with-open-file
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
		    (include		;<thread>
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
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	(out "cmake_minimum_required( VERSION 3.4 )")
	(out "project( mytest LANGUAGES CXX )")
	(out "set( CMAKE_CXX_COMPILER nvc++ )")
	(out "set( CMAKE_CXX_FLAGS \"-stdpar\"  )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set( CMAKE_CXX_STANDARD 17 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
		
					;(out "set( CMAKE_CXX_FLAGS )")
					;(out "find_package( Vulkan )")
	(out "set( SRCS ~{~a~^~%~} )"
	     (directory "source/*.cpp"))
	(out "add_executable( mytest ${SRCS} )")
	(out "target_include_directories( mytest PUBLIC /home/martin/stage/cl-cpp-generator2/example/58_stdpar/source/ )")
		
					;(out "target_link_libraries( mytest PRIVATE vulkan )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	)
      )))



