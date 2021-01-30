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
	`(surface ()
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
      ;; param ::= [<var-name> [types]*]*
      ;; types will be concatenated with space
      ;; if no types is given, the previous type is used
      ;; decl conveys arbitrary declarations like: ((construct (a 3))
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
	      `(defmethod ,name (,@(loop for param in params
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
		  ,(defuns
		      :defs
		       `((_CheckGL  ((f const char*)
					  (l int))
				 
				 )
		       (CreateVBO ((data const GLfloat*)
					   (size const uint))
				  :return GLuint)
		       (BindVBO  ((idx const uint)
					 (N)
					 (id cont GLuint)))
			 (CheckShader  ((shader GLuint)
					(vshader const char*)
					(fshader)))
			 (CheckProgram ((id GLuint)
					(vshader const char*)
					(fshader)))
			 (DrawQuad ()))
		    )
		  )))
     
    
    
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



