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
	       (_stdout_mutex :type "std::mutex"))
	      (do0
	       (include <iostream>
			<chrono>
			<thread>
			
					;<future>
					; <experimental/future>
						<pybind11/embed.h>
			)
	       " "
	       #+nil (include ; <boost/lexical_cast.hpp>
		
		)
	       " "
	       
	  

	    #+nil   ,(let ((l `((Exact_predicates_inexact_constructions_kernel nil K)
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

	     #+nil  (defun call_cgal_cpp ()
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
							,(format nil "v~a" f)))))
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


				   (do0
				    (CGAL--lloyd_optimize_mesh_2 cdt
								 "CGAL::parameters::max_iteration_number=10")
				    ,(logprint "after lloyd optimization"
					       `((cdt.number_of_vertices))))
				   ))))


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
		 
		 #+nil (call_cgal_cpp)
		  )

		     
		 #-nil
		 (progn
		   "pybind11::scoped_interpreter guard{};"
		   (pybind11--exec (string-r "
import sys
import IPython 
import cgal_mesher
print('hello world from PYTHON {}'.format(sys.version))
IPython.start_ipython()
")))
		      

		 (return 0)))))

    (define-module
       `(mesher_module ()
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

	    #+nil   (let ((state ,(emit-globals :init t)))
		 (declare (type "State" state)))
	       	       
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
		 (include <pybind11/pybind11.h>)
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
		 (include "vis_01_mesher_module.hpp")
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
					    ("CDT::Point" Point)
					   )
		       collect
		       (format nil "using ~a = ~a;" f e))
		     ))

	       "namespace py = pybind11;"
	       
	       "using namespace std::chrono_literals;"

	      
	       " "
	       (defclass+ (TypedInputIterator :template "typename T") ()
		 "public:"
		 ,@(loop for (e f) in `((iterator_category "std::input_iterator_tag")
					(difference_type "std::ptrdiff_t")
					(value_type T)
					(pointer T*)
					(reference T&)
					)
			 collect
			(format nil "using ~a = ~a;" e f))
		 #+nil (do0 "typedef T value_type;"
		      "typedef T& reference;"
		      "typedef T* pointer;")
		 (defmethod TypedInputIterator (py_iter)
		   (declare (type "py::iterator&" py_iter)
			    (construct (py_iter_ py_iter))
			    ;( explicit)  (:constructor)
			    (values explicit)))
		 (defmethod TypedInputIterator (py_iter)
		   (declare (type "py::iterator&&" py_iter)
			    (construct (py_iter_ py_iter))
			    (values explicit)))
		 (defmethod operator* ()
		   (declare (values value_type))
		   (return (dot "(*py_iter_)" ("template cast<value_type>"))))
		 (defmethod operator++ (inc)
		   (declare (type int inc)
			    (values TypedInputIterator))
		   (let ((copy *this))
		     ++py_iter_
		     (return copy)))
		 (defmethod operator++ ()
		   (declare 
			    (values TypedInputIterator&))
		   ++py_iter_
		   (return *this))
		 (defmethod operator!= (rhs)
		   (declare (type TypedInputIterator& rhs)
			    (values bool))
		   (return (!= py_iter_ rhs.py_iter_)))
		 (defmethod operator== (rhs)
		   (declare (type TypedInputIterator& rhs)
			    (values bool))
		   (return (== py_iter_ rhs.py_iter_)))
		 "private:"
		 "py::iterator py_iter_;")
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
	       (space PYBIND11_MODULE
		      (paren cgal_mesher m)
		      (progn
			(dot (py--class_<Point> m (string "Point"))
			     (def ("py::init<int,int>")
				 (py--arg (string "x"))
			       (py--arg (string "y")))
			     (def ("py::init<double,double>")
				 (py--arg (string "x"))
			       (py--arg (string "y")))
			     (def_property_readonly (string "x")  &Point--x)
			     (def_property_readonly (string "y")  &Point--y)
			     (def (string "__repr__")
				 (lambda (p)
				   (declare (type "const Point&" p))
				   (let ((r (std--string (string "Point("))))
				     (incf r (std--to_string (p.x)))
				     (incf r (string ", "))
				     (incf r (std--to_string (p.x)))
				     (incf r (string ")"))
				     (return r))))
			     (def (string "__eq__")
				 (lambda (p q)
				   (declare (type "const Point&" p q))
				   (return (== p q))))
			     (def (string "__hash__")
				 (lambda (p)
				   (declare (type "const Point&" p))
				   "std::hash<double> dh;"
				   (let ((xh (dh (p.x)))
					 (yh (dh (p.y))))
				     (return (+ (logxor yh xh
							)
						(hex #x9e3779b9)
						(<< yh 6)
						(>> yh 2)))))))
			(dot (py--class_<Vertex_handle> m (string "VertexHandle"))
			     (def_property_readonly
				 (string "point")
				 (lambda (handle)
				   (declare (type "Vertex_handle&" handle))
				   (return (handle->point)))))
			(dot (py--class_<CDT--Finite_vertices_iterator--value_type> m (string "Vertex"))
			     (def_property_readonly
				 (string "point")
				 (lambda (vertex)
				   (declare (type "CDT::Finite_vertices_iterator::value_type&" vertex))
				   (return (vertex.point)))))
			(dot (py--class_<CDT--Finite_faces_iterator--value_type> m (string "Face"))
			     (def
				 (string "vertex_handle")
				 (lambda (face index)
				   (declare (type "CDT::Finite_faces_iterator::value_type&" face)
					    (type int index))
				   (return (face.vertex index)))
			       (py--arg (string "index"))))
			(m.def (string "print_faces_iterator_value_type")
				    (lambda ()
				      (<< std--cout
					  (type_name<CDT--Finite_faces_iterator--value_type>)
					  std--endl)))
			(dot (py--class_<CDT> m (string "ConstrainedDelaunayTriangulation"))
			     
			     (def (py--init))
			     #-nil ,@(loop for e in `((insert ((CDT& cdt) ("const Point&" p)))
						      (insert_constraint ((CDT& cdt) (Vertex_handle a)
									  (Vertex_handle b)))
						      )
				     collect
				     (destructuring-bind (name params) e
				       `(def (string ,name)
					    (lambda ,(mapcar #'second params)
					      (declare ,@(loop for (type var) in params
							       collect
							       `(type ,type ,var)))
					      (return (dot ,(second (first params))
							   (,name ,@(mapcar #'second (cdr params)))))))))
			     ,@(loop for e in `(number_of_vertices
						number_of_faces)
				     collect
				     `(def (string ,e)
					  ,(format nil "&CDT::~a" e)))
			     
			     ,@(loop for e in `(finite_vertices finite_faces)
				     collect
				     `(def (string ,e)
					  (lambda (cdt)
					    (declare (type CDT& cdt)
						     (values "py::iterator"))
					    (return
					      (py--make_iterator
					       (dot
						cdt
						
						(,(format nil "~a_begin" e))
						)
					       (dot
						cdt
						
						(,(format nil "~a_end" e))
					)))))))

			(dot (py--class_<Criteria> m (string "Criteria"))
			     (def ("py::init<double,double>")
			       "py::arg(\"aspect_bound\")=0.125"
			       "py::arg(\"size_bound\")=0.0"
			       )
			     (def ("py::init<float,float>")
			       "py::arg(\"aspect_bound\")=0.125"
			       "py::arg(\"size_bound\")=0.0"
			       )
			     (def_property (string "size_bound")
			       "&Criteria::size_bound"
			       "&Criteria::set_size_bound")
			    (def_property (string "aspect_bound")
				 (lambda (c)
				   (declare (type "const Criteria&" c))
				   ;,(logprint "" `((c.bound)))
				   (return (c.bound)))
			       (lambda (c bound)
				 (declare (type "Criteria&" c)
					  (type double bound))
				 ;,(logprint "" `(bound))
				   (c.set_bound bound))))
			(dot (py--class_<Mesher> m (string "Mesher"))
			     (def (py--init<CDT&>))
			    (def (string "seeds_from")
				 (lambda (mesher iterable)
				   (declare (type "Mesher&" mesher)
					    (type py--iterable iterable))
				   (comments "cast the iterator to Point, otherwise it would assume python object type")
				   (let ((it (py--iter iterable))
					 (beg (TypedInputIterator<Point> it))
					 (end (TypedInputIterator<Point> (py--iterator--sentinel))))
				     (mesher.set_seeds beg end)
				     )))
			    (def (string "refine_mesh")
			      &Mesher--refine_mesh)
			    (def_property (string "criteria")
			      &Mesher--get_criteria
			      (lambda (mesher criteria)
				(declare (type Mesher& mesher)
					 (type "const Criteria&" criteria))
				(mesher.set_criteria criteria)))
			    )
			,@(loop for e in `((make_conforming_delaunay &CGAL--make_conforming_Delaunay_2<CDT> "Make a triangulation conform to the Delaunay property")
					   (make_conforming_gabriel &CGAL--make_conforming_Gabriel_2<CDT> "Make a triangulation conform to the Gabriel property"))
				collect
				(destructuring-bind (pyname cname doc) e
				  `(m.def (string ,pyname)
					  ,cname
					  (py--arg (string "cdt"))
					  (string ,doc))))
			(do0
			 (comments "boost named argument magic")
			 (m.def (string "lloyd_optimize")
				(lambda (cdt
					 max_iteration_number
					 time_limit
					 convergence
					 freeze_bound)
				  (declare (type CDT& cdt)
					   (type int max_iteration_number)
					   (type double time_limit convergence freeze_bound))
				  (CGAL--lloyd_optimize_mesh_2
				   cdt
				   ,@(loop for e in `(max_iteration_number
						      time_limit
						      convergence
						      freeze_bound)
					   collect
					   `(= ,(format nil "CGAL::parameters::~a"
							   e)
						  ,e))
				   )
				  )
				,@(loop for e in `((cdt)
						      (max_iteration_number 0)
						      (time_limit 0.0d0)
						      (convergence 0.001d0)
						      (freeze_bound 0.001d0))
					   collect
					   (destructuring-bind (name &optional value) e 
					     (if value
						 `(= (py--arg (string ,name)) ,value)
						 `(py--arg (string ,name))))))))))))
    
    
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
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	(out "cmake_minimum_required( VERSION 3.4 )")
	(out "project( mytest LANGUAGES CXX )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set( CMAKE_CXX_STANDARD 14 )")
	;(out "set( CMAKE_CXX_COMPILER clang++ )")
	(out "find_package( Python COMPONENTS Interpreter Development REQUIRED )")
	(out "find_package( pybind11 REQUIRED )")

	;; GMP MPFI
	(out "find_package( CGAL QUIET COMPONENTS Core )")
					;(out "set( CMAKE_CXX_FLAGS )")
	(out "set( SRCS ~{~a~^~%~} )" ;(directory "source/*.cpp")
	     `(vis_00_base.cpp))
	(out "add_executable( mytest ${SRCS} )")
	(out "target_link_libraries( mytest PRIVATE pybind11::embed gmp )")
	(out "pybind11_add_module( cgal_mesher vis_01_mesher_module.cpp )")
	(out "target_link_libraries( cgal_mesher PRIVATE gmp )")
	(out "target_precompile_headers( cgal_mesher PRIVATE vis_01_mesher_module.hpp )"))
      )))



