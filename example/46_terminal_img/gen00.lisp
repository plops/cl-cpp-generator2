(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)


(progn
  (defparameter *source-dir* #P"example/46_terminal_img/source/")
  
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

  (let*  ()
    
    
    
    (define-module
       `(base ((_main_version :type "std::string")
	       (_code_repository :type "std::string")
	       (_code_generation_time :type "std::string")
	       )
	      (do0
	       
	    
		    (include <iostream>
			     <chrono>
			     <thread>
			     <sys/mman.h>
			     <fcntl.h>
			     <unistd.h>
			     <array>
			     <bitset>
			     <cmath>
			     <map>
			     )


		    
		    "using namespace std::chrono_literals;"
		    " "

		     (split-header-and-code
		     (do0
		      "// header"

		      " "

		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp")
		      ))


	       
  
		(do0
		 "uint8_t *img;"
	

		 


		  (defclass CharData ()
		    "public:"
		    (defmethod CharData (codepoint)
		      (declare
		       (type int codepoint)
		       (construct (codePoint codepoint))
		       (values :constructor)))
		    "std::array<int,3> fgColor = std::array<int,3>{0,0,0};"
		   "std::array<int,3> bgColor = std::array<int,3>{0,0,0};"
					 "int codePoint;"
		    
					 ))

		(defun createCharData_simple (img w h x0 y0 codepoint pattern
					      )
		  (declare (type uint8_t* img)
			   (type int w h x0 y0 codepoint pattern)
			   (values CharData))
		  (let ((result (CharData codepoint))
			(fg_count (* 4 4))
			(bg_count (* 4 4))
			(mask (hex #x80000000)))
		    (declare (type "const int" fg_count bg_count))
		    (let ((*avg (result.fgColor.data)))
		     (dotimes (y 4)
		       (dotimes (x 4)
			 (dotimes (i 3)
			   (incf (aref avg i)
				 (aref img (+ i (* 3 (+ x0 x (* w (+ y0 y)))))))))))
		    (let ((*avg1 (result.bgColor.data)))
		     (dotimes (y 4)
		       (dotimes (x 4)
			 (dotimes (i 3)
			   (incf (aref avg1 i)
				 (aref img (+ i (* 3 (+ x0 x (* w (+ y0 y 4)))))))))))
		    (do0 (comments "average color for each bucket")
			 
			 (unless (== 0 bg_count)
			   (dotimes (i 3)
			     (setf (aref result.bgColor i)
				   (/ (aref result.bgColor i)
				      bg_count)
				   )))
			 (unless (== 0 fg_count)
			   (dotimes (i 3)
			     (setf (aref result.fgColor i)
				   (/ (aref result.fgColor i)
				      fg_count)
				   ))))
		    (return result)))
		

		(do0
		 (defun sqr (x)
		   (declare (values float)
			    (type float x))
		   (return (* x x)))
	
		 (defun clamp_byte (value)
		   (declare (inline)
			    (type int value)
			    (values int))
		   (if (< 0 value)
		       (if (< value 255)
			   (return value)
			   (return 255))
		       (return 0)))
		 (defun emit_color (r g b bg)
		   (declare (type int r g b)
			    (type bool bg)
			    )
		   (if bg
		       (<< std--cout (string
				      "\\x1b[48;2;")
			   r (string ";")
			   g (string ";")
			   b (string "m"))
		       (<< std--cout (string
				      "\\x1b[38;2;")
			   r (string ";")
			   g (string ";")
			   b (string "m")))
		)

		 

		 (defun emitCodepoint (codepoint)
		   (declare (type int codepoint)
			    )
		   (when (< codepoint 128)
		     (<< std--cout (static_cast<char> codepoint))
		     return)
		   (when (< codepoint 0x7ff)
		     (<< std--cout (static_cast<char> (logior #xc0 (>> codepoint 6))))
		     (<< std--cout (static_cast<char> (logior #x80 (logand codepoint #x3f))))
		     return)
		   (when (< codepoint 0xffff)
		     (<< std--cout (static_cast<char> (logior #xe0 (>> codepoint 12))))
		     (<< std--cout (static_cast<char> (logior #x80 (logand (>> codepoint 6) #x3f))))
		     (<< std--cout (static_cast<char> (logior #x80 (logand codepoint #x3f))))
		     return)
		   (when (< codepoint 0x10ffff)
		     (<< std--cout (static_cast<char> (logior #xf0 (>> codepoint 18))))
		     (<< std--cout (static_cast<char> (logior #x80 (logand (>> codepoint 12) #x3f))))
		     (<< std--cout (static_cast<char> (logior #x80 (logand (>> codepoint 6) #x3f))))
		     (<< std--cout (static_cast<char> (logior #x80 (logand codepoint #x3f))))
		     return)
		   (<< std--cerr (string "error")))
		 (defun emit_image (img w h)
		   (declare (type uint8_t* img)
			    (type int w h))
		   (let ((lastCharData (CharData 0)))
		     (for ((= "int y" 0)
			   (<= y (- h 8))
			   (incf y 8))
			  (for ((= "int x" 0)
				(<= x (- w 4))
				(incf x 4))
			       (let ((charData (createCharData_simple img w h x y (hex #x2584) (hex #x0000ffff))))
				 (when (or (== 0 x)
					   (!= charData.bgColor
					       lastCharData.bgColor))
				   (emit_color (aref charData.bgColor 0)
					       (aref charData.bgColor 1)
					       (aref charData.bgColor 2)
					       true))
				 (when (or (== 0 x)
					   (!= charData.fgColor
					       lastCharData.fgColor))
				   (emit_color (aref charData.fgColor 0)
					       (aref charData.fgColor 1)
					       (aref charData.fgColor 2)
					       false))
				 (emitCodepoint charData.codePoint)
				 (setf lastCharData charData)
				 
				 )
			       )
			  (<< std--cout (string "\\x1b[0m")
			      std--endl)
			  ))))

		    
		    (defun main (argc argv
				 )
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      ;,(logprint "start" `(argc (aref argv 0)))
		      (let ((fd (--open (string "img.raw")
					O_RDONLY))
			    ("const w" 512)
			    ("const h" 285)
			    (img (reinterpret_cast<uint8_t*>
				  (mmap nullptr
					(* w h 3)
					PROT_READ
					(logior MAP_FILE MAP_SHARED)
					fd 0))))
			(emit_image img w h)
			(munmap img (* w h 3)
				)
			(--close fd)
			)
		      (return 0)))))

    (define-module
       `(complex (
	       )
	      (do0
	       
	    
		    (include <iostream>
			     <chrono>
			     <thread>
			     <sys/mman.h>
			     <fcntl.h>
			     <unistd.h>
			     <array>
			     <bitset>
			     <cmath>
			     <map>
			     )


		    
		    "using namespace std::chrono_literals;"
		    " "

		     (split-header-and-code
		     (do0
		      "// header"

		      " "

		      )
		     (do0
		      "// implementation"
		      (include "vis_00_base.hpp")
		      ))


	       
  #+nil ,(let ((l `((#x00000000 #x00a0)
			       (#x0000000f #x2581) ;; lower 1/8
			       (#x000000ff #x2582)
			       (#x00000fff #x2583)
			       (#x0000ffff #x2584)
			       (#x000fffff #x2585)
			       (#x00ffffff #x2586)
			       (#x0fffffff #x2587)
			       (#xeeeeeeee #x258a)
			       (#xcccccccc #x258c)
			       (#x88888888 #x258e)
			       (#x0000cccc #x2596)
			       (#x00003333 #x2597)
			       (#xcccc0000 #x2598)
			       (#xcccc3333 #x259a)
			       (#x33330000 #x259d)
			       (#x3333cccc #x259e)
			       (#x3333ffff #x259f)
			       (#x000ff000 #x2501)
			       (#x66666666 #x2503)
			       (#x00077666 #x250f)
			       (#x000ee666 #x2513)
			       (#x66677000 #x2517)
			       (#x666ee000 #x251b)
			       (#x66677666 #x2523)
			       (#x666ee666 #x252b)
			       (#x000ff666 #x2533)
			       (#x666ff000 #x253b)
			       (#x666ff666 #x254b)
			       (#x000cc000 #x2578)
			       (#x00066000 #x2579)
			       (#x00033000 #x257a)
			       (#x00066000 #x257b)
			       (#x06600660 #x254f)
			       (#x000f0000 #x2500)
			       (#x0000f000 #x2500)
			       (#x44444444 #x2502)
			       (#x22222222 #x2502)
			       (#x000e0000 #x2574)
			       (#x0000e000 #x2574)
			       (#x44440000 #x2575)
			       (#x22220000 #x2575)
			       (#x00030000 #x2576)
			       (#x00003000 #x2576)
			       (#x00004444 #x2577)
			       (#x00002222 #x2577)
			       (#x44444444 #x23a2)
			       (#x22222222 #x23a5)
			       (#x0f000000 #x23ba)
			       (#x00f00000 #x23bb)
			       (#x00000f00 #x23bc)
			       (#x000000f0 #x23bd)
			       (#x00066000 #x25aa)))
			  )

		       `(do0
			 "// bla")

		      )
	

		(defun createCharData (img w h x0 y0 codepoint pattern)
		  (declare (type uint8_t* img)
			   (type int w h x0 y0 codepoint pattern)
			   (values CharData))
		  (let ((result (CharData codepoint))
			(fg_count 0)
			(bg_count 0)
			(mask (hex #x80000000)))
		    (dotimes (y 8)
		      (dotimes (x 4)
			(let ((avg))
			  (declare (type int* avg))
			  (if (logand pattern mask)
			      (do0 (setf avg (result.fgColor.data))
				   (incf fg_count))
			      (do0 (setf avg (result.bgColor.data))
				   (incf bg_count)))
			  (dotimes (i 3)
			    (incf (aref avg i)
				  (aref img (+ i (* 3 (+ x0 x (* w (+ y0 y)))))))

			    )
			  (setf mask (>> mask 1)))))
		    (comments "average color for each bucket")
		    (dotimes (i 3)
		      (unless (== 0 bg_count)
			(setf (aref result.bgColor i)
			      (/ (aref result.bgColor i)
				 bg_count)
			      ))
		      (unless (== 0 fg_count)
			(setf (aref result.fgColor i)
			      (/ (aref result.fgColor i)
				 fg_count)
			      )))
		    (return result)))
		

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
									      *source-dir* file))))
		   (with-open-file (sh fn-h
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
		     )
		   (sb-ext:run-program "/usr/bin/clang-format"
				       (list "-i"  (namestring fn-h)
				   
				   ))
		   )
		 

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


