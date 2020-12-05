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
  (defparameter *source-dir* #P"example/50_pybind_cgal/source/")
  
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
	       (_stdout_mutex :type "std::mutex"))
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			
					;<future>
					; <experimental/future>
					;	<pybind11/embed.h>
			)
	       " "
	       #+nil (include ; <boost/lexical_cast.hpp>
		
		)
	       " "
	       
	      #+nil ,(let ((l `((Exact_predicates_inexact_constructions_kernel nil K)
			   (Constrained_Delaunay_triangulation_2 "<K,Tds>" CDT)
			   
			   (Delaunay_mesh_face_base_2 <K> Fb)
			   (Delaunay_mesh_vertex_base_2 <K> Vb)
			   (nil "<Vb,Fb>" Tds Triangulation_data_structure_2)
			   
			   (Delaunay_mesher_2 "<CDT,Criteria>" Mesher)
			   (Delaunay_mesh_size_criteria_2 <CDT> Criteria)
			   
			   (Triangulation_conformer_2)
			   (lloyd_optimize_mesh_2))))
		  `(do0
		    ,@(remove-if
		       #'null
		       (loop for e in l
			     collect
			     (destructuring-bind (name &optional f g new-name) e
			       (when name
				 `(do0 (include ,(format nil "<CGAL/~a.h>" name))
				       " ")))))
		    ))

	       ,(let ((l `((Exact_predicates_inexact_constructions_kernel nil K)
			   (Delaunay_mesh_face_base_2 <K> Fb)
			   (Delaunay_mesh_vertex_base_2 <K> Vb)
			   (nil "<Vb,Fb>" Tds Triangulation_data_structure_2)
			   (Constrained_Delaunay_triangulation_2 "<K,Tds>" CDT) 
			   (Delaunay_mesh_size_criteria_2 <CDT> Criteria)
			   			     
			   (Delaunay_mesher_2 "<CDT,Criteria>" Mesher)
			   
			   
			   (Triangulation_conformer_2)
			   (lloyd_optimize_mesh_2)
			   )))
		  `(do0

		     (split-header-and-code
		(do0
		 "// header"
		 ,@(remove-if
		       #'null
		       (loop for e in l
			     collect
			     (destructuring-bind (name &optional f g new-name) e
			       (when name
				 `(do0 (include ,(format nil "<CGAL/~a.h>" name))
				       " ")))))
		 )
		(do0
		 "// implementation"
		 (include "vis_00_base.hpp")
		 ,@(remove-if #'null
				 (loop for e in l
				       collect
				       (destructuring-bind (name &optional template short new-name) e
					 (when short
					   (format nil "using ~a = CGAL::~a~a;"
						   short
						   (if name
						       name
						       new-name)
						   (if template
						       template
						       "")
						 
						   )))))
		 ))

		     ,@(loop for (e f) in `(("CDT::Vertex_handle" Vertex_handle)
				      ("CDT::Point" Point))
		       collect
		       (format nil "using ~a = ~a;" f e))
		     ))

	       
	       
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
		   ,(g `_code_repository) (string ,(format nil "https://github.com/plops/cl-cpp-generator2/tree/master/example/48_future"))
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

		 (let ((cdt (CDT)))
		   ,(let #+nil ((l `((a -4 0)
			       (b 0 -1)
			       (c 4 0)
			       (d 0 1)))
			  (l2 `((a b)
				(b c)
				(c d)
				(d a))))
		      #-nil ((l `((a 100 269)
				       (b 246 269)
				       (c 246 223)
				       (d 303 223)
				       (e 303 298)
				       (f 246 298)
				       (g 246 338)
				       (h 355 338)
				       (i 355 519)
				       (j 551 519)
				       (k 551 445)
				       (l 463 445)
				       (m 463 377)
				       (n 708 377)
				       (o 708 229)
				       (p 435 229)
				       (q 435 100)
				       (r 100 100)

				       (s 349 236)
				       (t 370 236)
				       (u 370 192)
				       (v 403 192)
				       (w 403 158)
				       (x 349 158)

				       (y 501 336)
				       (z 533 336)
				       (1 519 307)
				       (2 484 307)

				       ))
				  (l2 `((a b)
					(b c)
					(c d)
					(d e)
					(e f)
					(f g)
					(g h)
					(h i)
					(i j)
					(j k)
					(k l)
					(l m)
					(m n)
					(n o)
					(o p)
					(p q)
					(q r)
					(r a)

					(s t)
					(t u)
					(u v)
					(v w)
					(w x)
					(x s)

					(y z)
					(z 1)
					(1 2)
					(2 y))))
		      `(do0
			,@(loop for (name e f) in l
				collect
				`(let ((,(format nil "v~a" name)
					 (cdt.insert (Point ,e ,f))))))
			,@(loop for (e f) in l2
				collect
				`(cdt.insert_constraint ,(format nil "v~a" e)
							,(format nil "v~a" f))))))
		 (do0
		  (do0 
		   ,(logprint "before2"
			      `((cdt.number_of_vertices)))))
		 
		 #-nil  (let (((mesher cdt))
		        (seeds (std--vector<Point> (curly (Point 505 325)
				       (Point 379 172)))))
		   (declare (type Mesher (mesher cdt))
			    ;(type "std::vector<Point>" seeds)
			    )
		   #-nil (mesher.set_seeds (seeds.begin)
					   (seeds.end))
			  #-nil (do0
		  (comments "modify the triangulation to be more conforming by introducing steiner vertices on constrained edges")
		  (do0 (CGAL--make_conforming_Delaunay_2 cdt)
		       ,(logprint "after conforming delaunay"
				  `((cdt.number_of_vertices))))
		  (do0 (CGAL--make_conforming_Gabriel_2 cdt)
		       ,(logprint "after conforming gabriel"
				  `((cdt.number_of_vertices))))

		   (do0 #+nil (CGAL--refine_Delaunay_mesh_2 cdt (Criteria .125 .5))
			(mesher.set_criteria (Criteria .125 30))
			(mesher.refine_mesh)
			,(logprint "after meshing"
				  `((cdt.number_of_vertices))))
		  ))    
		 #+nil
		 (progn
		   "pybind11::scoped_interpreter guard{};"
		   (pybind11--exec (string-r "
import sys
import IPython 
print('hello world from PYTHON {}'.format(sys.version))
IPython.start_ipython()
")))
		      

		 (return 0)))))
    
    
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
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	(out "cmake_minimum_required( VERSION 3.4 )")
	(out "project( mytest )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set( CMAKE_CXX_STANDARD 14 )")

	(out "find_package( pybind11 REQUIRED )")

	;; GMP MPFI
	(out "find_package( CGAL QUIET COMPONENTS Core )")
					;(out "set( CMAKE_CXX_FLAGS )")
	(out "set( SRCS ~{~a~^~%~} )" (directory "source/*.cpp"))
	(out "add_executable( mytest ${SRCS} )")
	(out "target_link_libraries( mytest PRIVATE pybind11::embed gmp )")
	(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )"))
      )))



